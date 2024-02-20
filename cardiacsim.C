/* 
 * Solves the Panfilov model using an explicit numerical scheme.
 * Based on code orginally provided by Xing Cai, Simula Research Laboratory 
 * and reimplementation by Scott B. Baden, UCSD
 * 
 * Modified and  restructured by Didem Unat, Koc University
 *
 */
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <iomanip>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>
#include <omp.h>
using namespace std;


// Utilities
// 

// Timer
// Make successive calls and take a difference to get the elapsed time.
static const double kMicro = 1.0e-6;
double getTime()
{
    struct timeval TV;
    struct timezone TZ;

    const int RC = gettimeofday(&TV, &TZ);
    if(RC == -1) {
            cerr << "ERROR: Bad call to gettimeofday" << endl;
            return(-1);
    }

    return( ((double)TV.tv_sec) + kMicro * ((double)TV.tv_usec) );

}  // end getTime()

// Allocate a 2D array
double **alloc2D(int m,int n){
   double **E;
   int nx=n, ny=m;
   E = (double**)malloc(sizeof(double*)*ny + sizeof(double)*nx*ny);
   assert(E);
   int j;
   for(j=0;j<ny;j++) 
     E[j] = (double*)(E+ny) + j*nx;
   return(E);
}
    
// Reports statistics about the computation
// These values should not vary (except to within roundoff)
// when we use different numbers of  processes to solve the problem
 double stats(double **E, int m, int n, double *_mx){
     double mx = -1;
     double l2norm = 0;
     int i, j;
     for (j=1; j<=m; j++)
       for (i=1; i<=n; i++) {
          l2norm += E[j][i]*E[j][i];
          if (E[j][i] > mx)
              mx = E[j][i];
      }
     *_mx = mx;
     l2norm /= (double) ((m)*(n));
     l2norm = sqrt(l2norm);
     return l2norm;
 }

// External functions
extern "C" {
    void splot(double **E, double T, int niter, int m, int n);
}
void cmdLine(int argc, char *argv[], double& T, int& n, int& px, int& py, int& plot_freq, int& no_comm, int&num_threads);


void simulate (double** E,  double** E_prev,double** R,
	       const double alpha, const int n, const int m, const double kk,
	       const double dt, const double a, const double epsilon,
	       const double M1,const double  M2, const double b)
{
  int i, j; 
    /* 
     * Copy data from boundary of the computational box 
     * to the padding region, set up for differencing
     * on the boundary of the computational box
     * Using mirror boundaries
     */
  
    for (j=1; j<=m; j++) 
      E_prev[j][0] = E_prev[j][2];
    for (j=1; j<=m; j++) 
      E_prev[j][n+1] = E_prev[j][n-1];
    
    for (i=1; i<=n; i++) 
      E_prev[0][i] = E_prev[2][i];
    for (i=1; i<=n; i++) 
      E_prev[m+1][i] = E_prev[m-1][i];
    
    // Solve for the excitation, the PDE
    for (j=1; j<=m; j++){
      for (i=1; i<=n; i++) {
	E[j][i] = E_prev[j][i]+alpha*(E_prev[j][i+1]+E_prev[j][i-1]-4*E_prev[j][i]+E_prev[j+1][i]+E_prev[j-1][i]);
      }
    }
    
    /* 
     * Solve the ODE, advancing excitation and recovery to the
     *     next timtestep
     */
    for (j=1; j<=m; j++){
      for (i=1; i<=n; i++)
	      E[j][i] = E[j][i] -dt*(kk* E[j][i]*(E[j][i] - a)*(E[j][i]-1)+ E[j][i] *R[j][i]);
    }
    
    for (j=1; j<=m; j++){
      for (i=1; i<=n; i++)
	      R[j][i] = R[j][i] + dt*(epsilon+M1* R[j][i]/( E[j][i]+M2))*(-R[j][i]-kk * E[j][i]*(E[j][i]-b-1));
    }
    
}

void parallel1D(double** local_E, double** local_E_prev, double** local_R,
                   const double alpha, const int n, const int local_m, const double kk,
                   const double dt, const double a, const double epsilon,
                   const double M1, const double M2, const double b, int world_rank, int world_size, int num_threads) {
    MPI_Request requests[4];
    int num_requests = 0;
    int up = world_rank - 1;
    int down = world_rank + 1;

    // For handling edge cases
    if (up < 0) up = MPI_PROC_NULL;
    if (down >= world_size) down = MPI_PROC_NULL;
   
    // Posting data for all neighbors
    if (up != MPI_PROC_NULL) {
        MPI_Irecv(local_E_prev[0], n + 2, MPI_DOUBLE, up, 0, MPI_COMM_WORLD, &requests[num_requests++]);
    }
    if (down != MPI_PROC_NULL) {
        MPI_Irecv(local_E_prev[local_m + 1], n + 2, MPI_DOUBLE, down, 1, MPI_COMM_WORLD, &requests[num_requests++]);
    }

    // Sending data to neighbors
    if (up != MPI_PROC_NULL) {
        MPI_Isend(local_E_prev[1], n + 2, MPI_DOUBLE, up, 1, MPI_COMM_WORLD, &requests[num_requests++]);
    }
    if (down != MPI_PROC_NULL) {
        MPI_Isend(local_E_prev[local_m], n + 2, MPI_DOUBLE, down, 0, MPI_COMM_WORLD, &requests[num_requests++]);
    }

    // Waiting for all MPI communications
    MPI_Waitall(num_requests, requests, MPI_STATUSES_IGNORE);

    // Applying boundary conditions
    for (int j = 1; j <= local_m; j++) {
        local_E_prev[j][0] = local_E_prev[j][2];
        local_E_prev[j][n+1] = local_E_prev[j][n-1];
    }

    if (up == MPI_PROC_NULL) {
        for (int i = 1; i <= n; i++) {
            local_E_prev[0][i] = local_E_prev[2][i];  // Top boundary
        }
    }

    if (down == MPI_PROC_NULL) {
        for (int i = 1; i <= n; i++) {
            local_E_prev[local_m+1][i] = local_E_prev[local_m-1][i];  // Bottom boundary
        }
    }
     #pragma omp parallel for collapse(2) num_threads(num_threads)
    for (int j = 1; j <= local_m; j++) {
        for (int i = 1; i <= n; i++) {
            local_E[j][i] = local_E_prev[j][i] + alpha * (local_E_prev[j][i+1] + local_E_prev[j][i-1] - 4 * local_E_prev[j][i] + local_E_prev[j+1][i] + local_E_prev[j-1][i]);
        }
    }
     #pragma omp parallel for collapse(2) num_threads(num_threads)
    for (int j = 1; j <= local_m; j++) {
        for (int i = 1; i <= n; i++) {
            local_E[j][i] = local_E[j][i] - dt * (kk * local_E[j][i] * (local_E[j][i] - a) * (local_E[j][i] - 1) + local_E[j][i] * local_R[j][i]);
        }
    }
     #pragma omp parallel for collapse(2) num_threads(num_threads)
    for (int j = 1; j <= local_m; j++) {
        for (int i = 1; i <= n; i++) {
           
            local_R[j][i] = local_R[j][i] + dt * (epsilon + M1 * local_R[j][i] / (local_E[j][i] + M2)) * (-local_R[j][i] - kk * local_E[j][i] * (local_E[j][i] - b - 1));
        }
    }
    
    
}


// Main program
int main (int argc, char** argv)
{
  /*
   *  Solution arrays
   *   E is the "Excitation" variable, a voltage
   *   R is the "Recovery" variable
   *   E_prev is the Excitation variable for the previous timestep,
   *      and is used in time integration
   */
  double **E, **R, **E_prev;
  
  // Various constants - these definitions shouldn't change
  const double a=0.1, b=0.1, kk=8.0, M1= 0.07, M2=0.3, epsilon=0.01, d=5e-5;
  
  double T=1000.0;
  int m=200,n=200;
  int plot_freq = 0;
  int px = 1, py = 0;
  int no_comm = 0;
  int num_threads=1; 




  MPI_Init(&argc, &argv);   
  int world_size;         //number of processes
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);



  cmdLine( argc, argv, T, n,px, py, plot_freq, no_comm, num_threads);
  m = n;  
  // Allocate contiguous memory for solution arrays
  // The computational box is defined on [1:m+1,1:n+1]
  // We pad the arrays in order to facilitate differencing on the 
  // boundaries of the computation box
  if(world_rank==0){
    E = alloc2D(m+2,n+2);
   E_prev = alloc2D(m+2,n+2);
   R = alloc2D(m+2,n+2);
  }

  int rows_per_process = m / world_size;
  int extra_rows = m % world_size;
  int local_m = (world_rank == world_size - 1) ? (rows_per_process + extra_rows) : rows_per_process;

  double **local_E = alloc2D(local_m +2, n + 2);
  double **local_E_prev = alloc2D(local_m + 2, n + 2);
  double **local_R = alloc2D(local_m + 2, n + 2); 




  for (int j = 1; j <= local_m; j++) {
    for (int i = 1; i <= n; i++) {
      local_E_prev[j][i] = 0;
      local_R[j][i] = 0;
    }
  }

  int global_start_row = world_rank * rows_per_process + 1;
  for (int j = 1; j <= local_m; j++) {
    for (int i = n / 2 + 1; i <= n; i++) {
      local_E_prev[j][i] = 1.0;
    }
    if (global_start_row + j - 1 >= m / 2 + 1) {
      for (int i = 1; i <= n; i++) {
        local_R[j][i] = 1.0;
      }
    }
  }


  if(world_rank==0){
  int i,j;
  // Initialization
  for (j=1; j<=m; j++)
    for (i=1; i<=n; i++)
      E_prev[j][i] = R[j][i] = 0;
  
  for (j=1; j<=m; j++)
    for (i=n/2+1; i<=n; i++)
      E_prev[j][i] = 1.0;
  
  for (j=m/2+1; j<=m; j++)
    for (i=1; i<=n; i++)
      R[j][i] = 1.0;
  }
  
  double dx = 1.0/n;

  // For time integration, these values shouldn't change 
  double rp= kk*(b+1)*(b+1)/4;
  double dte=(dx*dx)/(d*4+((dx*dx))*(rp+kk));
  double dtr=1/(epsilon+((M1/M2)*rp));
  double dt = (dte<dtr) ? 0.95*dte : 0.95*dtr;
  double alpha = d*dt/(dx*dx);

  printf("%d numthreads\n", num_threads); 

  cout << "aaaa aaaa: " << num_threads << " x " << endl;

  cout << "Grid Size       : " << n << endl; 
  cout << "Duration of Sim : " << T << endl; 
  cout << "Time step dt    : " << dt << endl; 
  cout << "Process geometry: " << px << " x " << py << endl;
  if (no_comm)
    cout << "Communication   : DISABLED" << endl;
  
  cout << endl;
  
  // Start the timer
  double t0 = getTime();
  
 
  // Simulated time is different from the integer timestep number
  // Simulated time
  double t = 0.0;
  // Integer timestep number
  int niter=0;
  
  while (t<T) {
    
    t += dt;
    niter++;
 
    if (py == 0){
      simulate(E, E_prev, R, alpha, n, m, kk, dt, a, epsilon, M1, M2, b);
      double **tmp = E; E = E_prev; E_prev = tmp;
    }
    else{
      parallel1D(local_E, local_E_prev, local_R, alpha, n, local_m, kk, dt, a, epsilon, M1, M2, b, world_rank, world_size, num_threads);
      double **tmpp = local_E; local_E = local_E_prev; local_E_prev = tmpp;

      if (world_rank == 0) {
            int *recvcounts = new int[world_size];
            int *displs = new int[world_size];

            // Calculate recvcounts and displacements
            for (int i = 0; i < world_size; i++) {
                recvcounts[i] = (i == world_size - 1) ? ((rows_per_process + extra_rows) * (n + 2)) : (rows_per_process * (n + 2));
                displs[i] = i * rows_per_process * (n + 2);
            }

            MPI_Gatherv(&(local_E[0][0]), local_m * (n + 2), MPI_DOUBLE, &(E[0][0]), recvcounts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            delete[] recvcounts;
            delete[] displs;
        } else {
            MPI_Gatherv(&(local_E[0][0]), local_m * (n + 2), MPI_DOUBLE, nullptr, nullptr, nullptr, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }
    } 
    
    if (plot_freq && world_rank == 0) {
        int k = (int)(t / plot_freq);
        if ((t - k * plot_freq) < dt) {
            splot(E, t, niter, m + 2, n + 2);
        }
    }
  
  }

   if (world_rank == 0) {

    // Post-simulation
    double time_elapsed = getTime() - t0;
    double Gflops = (double)(niter * (1E-9 * n * n ) * 28.0) / time_elapsed ;
    double BW = (double)(niter * 1E-9 * (n * n * sizeof(double) * 4.0  ))/time_elapsed;

    cout << "Number of Iterations        : " << niter << endl;
    cout << "Elapsed Time (sec)          : " << time_elapsed << endl;
    cout << "Sustained Gflops Rate       : " << Gflops << endl; 
    cout << "Sustained Bandwidth (GB/sec): " << BW << endl << endl; 

    double mx;
    double l2norm = stats(E,m,n,&mx);
    cout << "Max: " << mx <<  " L2norm: "<< l2norm << endl;

    if (plot_freq){
      cout << "\n\nEnter any input to close the program and the plot..." << endl;
      getchar();
    }

    // Clean up
     
    free(E);
    free(E_prev);
    free(R);
    }

    free(local_E);
    free(local_E_prev);
    free(local_R);
    MPI_Finalize();
    return 0;
}