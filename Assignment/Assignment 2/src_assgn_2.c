#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include "mpi.h"

//  mpirun -np P -f hostfile ./halo Px N <num_time_steps> <seed> <stencil>
//  #pragma prutor-mpi-args: -np 12 -ppn 6
//  #pragma prutor-mpi-sysargs: 4 25 5 7 9
//  TC-2 : #pragma prutor-mpi-sysargs: 4 262144 10 7 9
//  TC-3 : #pragma prutor-mpi-sysargs: 4 4194304 10 7 5
//  TC-4 : #pragma prutor-mpi-sysargs: 4 4194304 10 7 9

int non_leader(int Px, int data_size, int num_time_steps, int Seed, int myrank, int size)
{
    // Initialising loop control variables
    int i = 0;
    int j = 0;
    int l = 0;
    int dr = 0;
    int dc = 0;

    // The number of data points per processor
    int N = sqrt(data_size);

    int px = Px;
    int py = size / px;

    // flag = 1 for 5-stencil
    // flag = 2 for 9-stencil
    // Helpful for compact calculations
    int flag = 2;

    // The main buffer storing result at every timestep
    // Temporary buffer for stencil calculation
    // double buf[N][N], tmp[N][N];
    double * buf=malloc((N*N)*sizeof(double));
    double * tmp=malloc((N*N)*sizeof(double));

    // Defining buffers for message passing
    // First 2 rows used as send_bufs
    // Last 2 rows used as recv_bufs
    double top[4][N];
    double bot[4][N];
    double right[4][N];
    double left[4][N];

    // Initialising data according to question
    int seed = Seed;
    srand(seed * (myrank + 10));
    
    for(i = 0; i < N; i++)
    {
        for(j = 0; j < N; j++)
        {
            buf[i*N+j] = abs(rand() + (i * rand() + j * myrank)) / 100;
        }
    }

    // (row, col) in the logical processor matrix
    int row = myrank / px;
    int col = myrank % px;
    int tm = num_time_steps;

    // Noting the starting time
    double t1, t2;
    t1 = MPI_Wtime();

    // Loop for timesteps
    for (l = 0; l < tm; l++)
    {
        // Calling Isend/Irecv for all sides

        // Buffer string all the MPI_Request objects and the count of requests
        MPI_Request request_buf[16];
        int count = 0;

        for (dr = -1; dr <= 1; dr++)
        {
            for (dc = -1; dc <= 1; dc++)
            {
                // Checking valid cases for communication (top, bottom, right, left)
                if (abs(dr) != abs(dc) && row + dr >= 0 && row + dr < py && col + dc >= 0 && col + dc < px)
                {

                    // (row,col) of neighbours with which we are communicating
                    int nr = row + dr;
                    int nc = col + dc;

                    // Sending and Recieving with bottom neighbour
                    // Packing data in bot[0] and bot[1] (for stencil size 9)
                    if (nr > row)
                    {
                        for (i = 0; i < flag; i++)
                        {
                            int position = 0;
                            for (j = 0; j < N; j++)
                            {
                                MPI_Pack(&buf[(N-1-i)*N+j], 1, MPI_DOUBLE, bot[i], N * 8, &position, MPI_COMM_WORLD);
                            }
                            MPI_Isend(bot[i], position, MPI_PACKED, nr * px + nc, nr * px + nc, MPI_COMM_WORLD, &request_buf[count]);
                            count++;
                            MPI_Irecv(&bot[i + 2][0], N, MPI_DOUBLE, nr * px + nc, row * px + col, MPI_COMM_WORLD, &request_buf[count]);
                            count++;
                        }
                    }

                    // Sending and Recieving with top neighbour
                    // Packing data in top[0] and top[1] (for stencil size 9)
                    if (nr < row)
                    {
                        for (i = 0; i < flag; i++)
                        {
                            int position = 0;
                            for (j = 0; j < N; j++)
                            {
                                MPI_Pack(&buf[i*N+j], 1, MPI_DOUBLE, top[i], N * 8, &position, MPI_COMM_WORLD);
                            }
                            MPI_Isend(top[i], position, MPI_PACKED, nr * px + nc, nr * px + nc, MPI_COMM_WORLD, &request_buf[count]);
                            count++;
                            MPI_Irecv(top[i + 2], N, MPI_DOUBLE, nr * px + nc, row * px + col, MPI_COMM_WORLD, &request_buf[count]);
                            count++;
                        }
                    }

                    // Sending and Recieving with right neighbour
                    // Packing data in right[0] and right[1] (for stencil size 9)
                    if (nc > col)
                    {
                        for (i = 0; i < flag; i++)
                        {
                            int position = 0;
                            for (j = 0; j < N; j++)
                            {
                                MPI_Pack(&buf[j*N+N-1-i], 1, MPI_DOUBLE, right[i], N * 8, &position, MPI_COMM_WORLD);
                            }
                            MPI_Isend(right[i], position, MPI_PACKED, nr * px + nc, nr * px + nc, MPI_COMM_WORLD, &request_buf[count]);
                            count++;
                            MPI_Irecv(right[i + 2], N, MPI_DOUBLE, nr * px + nc, row * px + col, MPI_COMM_WORLD, &request_buf[count]);
                            count++;
                        }
                    }

                    // Sending and Recieving with left neighbour
                    // Packing data in left[0] and left[1] (for stencil size 9)
                    if (nc < col)
                    {
                        for (i = 0; i < flag; i++)
                        {
                            int position = 0;
                            for (j = 0; j < N; j++)
                            {
                                MPI_Pack(&buf[j*N+i], 1, MPI_DOUBLE, left[i], N * 8, &position, MPI_COMM_WORLD);
                            }
                            MPI_Isend(left[i], position, MPI_PACKED, nr * px + nc, nr * px + nc, MPI_COMM_WORLD, &request_buf[count]);
                            count++;
                            MPI_Irecv(left[i + 2], N, MPI_DOUBLE, nr * px + nc, row * px + col, MPI_COMM_WORLD, &request_buf[count]);
                            count++;
                        }
                    }
                }
            }
        }

        // Waiting for all the Isend's and Irecv's to complete before unpacking
        MPI_Status status_buf[count];
        MPI_Waitall(count, request_buf, status_buf);

        // Unpacking all the received data
        for (dr = -1; dr <= 1; dr++)
        {
            for (dc = -1; dc <= 1; dc++)
            {
                // Checking which of top, right, bottom or left buffers need to be unpacked
                if (abs(dr) != abs(dc) && row + dr >= 0 && row + dr < py && col + dc >= 0 && col + dc < px)
                {

                    // (row,col) of neighbours with which we have communicated
                    int nr = row + dr;
                    int nc = col + dc;

                    // Unpacking bot[2] and bot[3] into bot[0] and bot[1]
                    if (nr > row)
                    {
                        for (i = 0; i < flag; i++)
                        {
                            int position = 0;
                            for (j = 0; j < N; j++)
                            {
                                MPI_Unpack(bot[i + 2], 8 * N, &position, &bot[i][j], 1, MPI_DOUBLE, MPI_COMM_WORLD);
                            }
                        }
                    }

                    // Unpacking top[2] and top[3] into top[0] and top[1]
                    if (nr < row)
                    {
                        for (i = 0; i < flag; i++)
                        {
                            int position = 0;
                            for (j = 0; j < N; j++)
                            {
                                MPI_Unpack(top[i + 2], 8 * N, &position, &top[i][j], 1, MPI_DOUBLE, MPI_COMM_WORLD);
                            }
                        }
                    }

                    // Unpacking right[2] and right[3] into right[0] and right[1]
                    if (nc > col)
                    {
                        for (i = 0; i < flag; i++)
                        {
                            int position = 0;
                            for (j = 0; j < N; j++)
                            {
                                MPI_Unpack(right[i + 2], 8 * N, &position, &right[i][j], 1, MPI_DOUBLE, MPI_COMM_WORLD);
                            }
                        }
                    }

                    // Unpacking left[2] and left[3] into left[0] and left[1]
                    if (nc < col)
                    {
                        for (i = 0; i < flag; i++)
                        {
                            int position = 0;
                            for (j = 0; j < N; j++)
                            {
                                MPI_Unpack(left[i + 2], 8 * N, &position, &left[i][j], 1, MPI_DOUBLE, MPI_COMM_WORLD);
                            }
                        }
                    }
                }
            }
        }

        // Performing the stencil calculation for both stencil sizes
        for (i = 0; i < N; i++)
        {
            for (j = 0; j < N; j++)
            {

                int nn = 1;            // Number of values average is calculated on
                tmp[i*N+j] = buf[i*N+j]; // Filling the temporary buffer with new values

                int bfi, ti, k;
                // bfi tracks index in buffer
                // ti tracks index in send/recv buffers
                // k tracks the distance of stencil point from center

                // bottom computation if exist
                bfi = i + 1, ti = 0, k = 0;
                while (bfi < N && k < flag)
                    tmp[i*N+j] += buf[bfi*N+j], bfi++, k++, nn++;
                if ((myrank / px) != py - 1)
                {
                    while (k < flag)
                        tmp[i*N+j] += bot[ti][j], ti++, k++, nn++;
                }

                // top computation if exist
                bfi = i - 1, ti = 0, k = 0;
                while (bfi >= 0 && k < flag)
                    tmp[i*N+j] += buf[bfi*N+j], bfi--, k++, nn++;
                if ((myrank / px) != 0)
                {
                    while (k < flag)
                        tmp[i*N+j] += top[ti][j], ti++, k++, nn++;
                }

                // right computation if exist
                bfi = j + 1, ti = 0, k = 0;
                while (bfi < N && k < flag)
                    tmp[i*N+j] += buf[i*N+bfi], bfi++, k++, nn++;
                if ((myrank % px) != px - 1)
                {
                    while (k < flag)
                        tmp[i*N+j] += right[ti][i], ti++, k++, nn++;
                }

                // left computation if exist
                bfi = j - 1, ti = 0, k = 0;
                while (bfi >= 0 && k < flag)
                    tmp[i*N+j] += buf[i*N+bfi], bfi--, k++, nn++;
                if ((myrank % px) != 0)
                {
                    while (k < flag)
                        tmp[i*N+j] += left[ti][i], ti++, k++, nn++;
                }

                // taking average
                tmp[i*N+j] = (tmp[i*N+j] / (double)(nn + 0.0));
            }
        }

        // Updating values of buffer from temporary buffer
        for (i = 0; i < N; i++)
        {
            for (j = 0; j < N; j++)
                buf[i*N+j] = tmp[i*N+j];
        }
    }

    // Noting the end time after computation + communication
    t2 = MPI_Wtime();

    double ans;
    double ft = t2 - t1;

    MPI_Reduce(&ft, &ans, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (myrank == 0)
    {
        printf("Time without leader = %lf\n", ans);
        printf("Data= %lf\n", buf[0]);
    }

    free(buf);
    free(tmp);

    return 0;
}

int leader(int Px, int data_size, int num_time_steps, int Seed, int myrank, int size)
{
    // Initialising loop control variables
    int i = 0;
    int j = 0;
    int l = 0;
    int dr = 0;
    int dc = 0;

    // The number of data points per processor
    int N = sqrt(data_size);

    int px = Px;
    int py = size / px;

    // The leader for the process
    int myleader = (myrank / px) * px;

    // Creating the communicators on the basis of common leaders
    MPI_Comm node_comm;
    MPI_Comm_split(MPI_COMM_WORLD, myleader, myrank, &node_comm);

    // Creating custom datatypes for easier intranode communication
    MPI_Datatype top_vector, bottom_vector;
    MPI_Type_vector(2, N, N, MPI_DOUBLE, &top_vector);
    MPI_Type_vector(2, N, -N, MPI_DOUBLE, &bottom_vector);
    MPI_Type_commit(&top_vector);
    MPI_Type_commit(&bottom_vector);

    // flag = 1 for 5-stencil
    // flag = 2 for 9-stencil
    // Helpful for compact calculations
    int flag = 2;

    // The main buffer storing result at every timestep
    // Temporary buffer for stencil calculation
    // double buf[N][N], tmp[N][N];
    double * buf=malloc((N*N)*sizeof(double));
    double * tmp=malloc((N*N)*sizeof(double));

    // Defining buffers for message passing
    // First 2 rows used as send_bufs
    // Last 2 rows used as recv_bufs
    double top[4][N];
    double bot[4][N];
    double right[4][N];
    double left[4][N];

    // Buffers for communication among leaders
    int r, c;
    if (myrank == myleader)
    {
        r = 2 * px;
        c = N;
    }
    else
        r = 0, c = 0;

    double top_send[r][c];
    double top_recv[r][c];
    double bot_send[r][c];
    double bot_recv[r][c];

    // Initialising data according to question
    int seed = Seed;
    srand(seed * (myrank + 10));

    for (i = 0; i < N; i++)
    {
        for (j = 0; j < N; j++)
        {
            buf[i*N+j] = abs(rand() + (i * rand() + j * myrank)) / 100;
        }
    }

    // (row, col) in the logical processor matrix
    int row = myrank / px;
    int col = myrank % px;
    int tm = num_time_steps;

    // Noting the starting time
    double t1, t2;
    t1 = MPI_Wtime();

    // Loop for timesteps
    for (l = 0; l < tm; l++)
    {
        //// Step-1-Calling Isend/Irecv for left and right neighbours

        // Buffer string all the MPI_Request objects and the count of requests
        MPI_Request request_buf[128];
        MPI_Status status_buf[128];
        int test_flag;
        int count = 0;

        for (dr = -1; dr <= 1; dr++)
        {
            for (dc = -1; dc <= 1; dc++)
            {
                // Checking valid cases for communication (top, bottom, right, left)
                if (abs(dr) != abs(dc) && row + dr >= 0 && row + dr < py && col + dc >= 0 && col + dc < px)
                {

                    // (row,col) of neighbours with which we are communicating
                    int nr = row + dr;
                    int nc = col + dc;

                    // Sending and Recieving with right neighbour
                    // Packing data in right[0] and right[1]
                    if (nc > col)
                    {
                        for (i = 0; i < flag; i++)
                        {
                            int position = 0;
                            for (j = 0; j < N; j++)
                            {
                                MPI_Pack(&buf[j*N+N - 1 - i], 1, MPI_DOUBLE, right[i], N * 8, &position, MPI_COMM_WORLD);
                            }
                            MPI_Isend(right[i], position, MPI_PACKED, nr * px + nc, nr * px + nc, MPI_COMM_WORLD, &request_buf[count]);
                            count++;
                            MPI_Irecv(right[i + 2], N, MPI_DOUBLE, nr * px + nc, row * px + col, MPI_COMM_WORLD, &request_buf[count]);
                            count++;
                        }
                    }

                    // Sending and Recieving with left neighbour
                    // Packing data in left[0] and left[1]
                    if (nc < col)
                    {
                        for (i = 0; i < flag; i++)
                        {
                            int position = 0;
                            for (j = 0; j < N; j++)
                            {
                                MPI_Pack(&buf[j*N+i], 1, MPI_DOUBLE, left[i], N * 8, &position, MPI_COMM_WORLD);
                            }
                            MPI_Isend(left[i], position, MPI_PACKED, nr * px + nc, nr * px + nc, MPI_COMM_WORLD, &request_buf[count]);
                            count++;
                            MPI_Irecv(left[i + 2], N, MPI_DOUBLE, nr * px + nc, row * px + col, MPI_COMM_WORLD, &request_buf[count]);
                            count++;
                        }
                    }
                }
            }
        }

        MPI_Waitall(count, request_buf, status_buf);
        count = 0;

        //// Step-2-Gathering all the top and bottom values in the intranode leader 

        // Need not gather top values of top-most processes
        if (myrank / px > 0)
        {
            MPI_Gather(&buf[0],1,top_vector,top_send[0],2*N,MPI_DOUBLE,0,node_comm);
        }

        // Need not gather bottom values of bottom-most processes
        if ((myrank / px) < (py - 1))
        {
            MPI_Gather(&buf[N-1],1,bottom_vector,bot_send[0],2*N,MPI_DOUBLE,0,node_comm);
        }

        count = 0;
        
        //// Step-3-Internode communication - 1D nearest neighbour communication

        // Only for leader ranks
        if (myrank == myleader)
        {   
            // Sending and receiving from bottom neighbour
            if ((myrank / px) < (py - 1))
            {
                MPI_Isend(bot_send[0], 2 * px * N, MPI_DOUBLE, myrank + px, myrank + px, MPI_COMM_WORLD, &request_buf[count]);
                count++;
                MPI_Irecv(bot_recv[0], 2 * px * N, MPI_DOUBLE, myrank + px, myrank, MPI_COMM_WORLD, &request_buf[count]);
                count++;
            }

            // Sending and receiving from top neighbour
            if ((myrank / px) > 0)
            {
                MPI_Isend(top_send[0], 2 * px * N, MPI_DOUBLE, myrank - px, myrank - px, MPI_COMM_WORLD, &request_buf[count]);
                count++;
                MPI_Irecv(top_recv[0], 2 * px * N, MPI_DOUBLE, myrank - px, myrank, MPI_COMM_WORLD, &request_buf[count]);
                count++;
            }

            // Waiting for all asynchronous calls
            MPI_Waitall(count, request_buf, status_buf);
        }

        
        count = 0;

        //// Step-4-Scattering all the top and bottom values too all the processes on the same node

        // Need not scatter top values of top-most processes
        if (myrank / px > 0)
        {
            MPI_Scatter(top_recv[0],2*N,MPI_DOUBLE,top[0],2*N,MPI_DOUBLE,0,node_comm);
        }

        // Need not scatter bottom values of bottom-most processes
        if ((myrank / px) < (py - 1))
        {
            MPI_Scatter(bot_recv[0],2*N,MPI_DOUBLE,bot[0],2*N,MPI_DOUBLE,0,node_comm);
        }

        //// Step-5-Unpacking the data from left and right neighbourhood communications

        // Unpacking all the received data
        for (dr = -1; dr <= 1; dr++)
        {
            for (dc = -1; dc <= 1; dc++)
            {
                // Checking which of top, right, bottom or left buffers need to be unpacked
                if (abs(dr) != abs(dc) && row + dr >= 0 && row + dr < py && col + dc >= 0 && col + dc < px)
                {

                    // (row,col) of neighbours with which we have communicated
                    int nr = row + dr;
                    int nc = col + dc;

                    // Unpacking right[2] and right[3] into right[0] and right[1]
                    if (nc > col)
                    {
                        for (i = 0; i < flag; i++)
                        {
                            int position = 0;
                            for (j = 0; j < N; j++)
                            {
                                MPI_Unpack(right[i + 2], 8 * N, &position, &right[i][j], 1, MPI_DOUBLE, MPI_COMM_WORLD);
                            }
                        }
                    }

                    // Unpacking left[2] and left[3] into left[0] and left[1]
                    if (nc < col)
                    {
                        for (i = 0; i < flag; i++)
                        {
                            int position = 0;
                            for (j = 0; j < N; j++)
                            {
                                MPI_Unpack(left[i + 2], 8 * N, &position, &left[i][j], 1, MPI_DOUBLE, MPI_COMM_WORLD);
                            }
                        }
                    }
                }
            }
        }

        //// Step-6-Performing the stencil calculations

        // Performing the stencil calculation
        for (i = 0; i < N; i++)
        {
            for (j = 0; j < N; j++)
            {

                int nn = 1;            // Number of values average is calculated on
                tmp[i*N+j] = buf[i*N+j]; // Filling the temporary buffer with new values

                int bfi, ti, k;
                // bfi tracks index in buffer
                // ti tracks index in send/recv buffers
                // k tracks the distance of stencil point from center

                // bottom computation if exist
                bfi = i + 1, ti = 0, k = 0;
                while (bfi < N && k < flag)
                    tmp[i*N+j] += buf[bfi*N+j], bfi++, k++, nn++;
                if ((myrank / px) != py - 1)
                {
                    while (k < flag)
                        tmp[i*N+j] += bot[ti][j], ti++, k++, nn++;
                }

                // top computation if exist
                bfi = i - 1, ti = 0, k = 0;
                while (bfi >= 0 && k < flag)
                    tmp[i*N+j] += buf[bfi*N+j], bfi--, k++, nn++;
                if ((myrank / px) != 0)
                {
                    while (k < flag)
                        tmp[i*N+j] += top[ti][j], ti++, k++, nn++;
                }

                // right computation if exist
                bfi = j + 1, ti = 0, k = 0;
                while (bfi < N && k < flag)
                    tmp[i*N+j] += buf[i*N+bfi], bfi++, k++, nn++;
                if ((myrank % px) != px - 1)
                {
                    while (k < flag)
                        tmp[i*N+j] += right[ti][i], ti++, k++, nn++;
                }

                // left computation if exist
                bfi = j - 1, ti = 0, k = 0;
                while (bfi >= 0 && k < flag)
                    tmp[i*N+j] += buf[i*N+bfi], bfi--, k++, nn++;
                if ((myrank % px) != 0)
                {
                    while (k < flag)
                        tmp[i*N+j] += left[ti][i], ti++, k++, nn++;
                }

                // taking average
                tmp[i*N+j] = (tmp[i*N+j] / (double)(nn + 0.0));
            }
        }

        // Updating values of buffer from temporary buffer
        for (i = 0; i < N; i++)
        {
            for (j = 0; j < N; j++)
                buf[i*N+j] = tmp[i*N+j];
        }
    }

    // Noting the end time after computation + communication
    t2 = MPI_Wtime();

    double ans;
    double ft = t2 - t1;

    // Getting the largest time on any processor
    MPI_Reduce(&ft, &ans, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (myrank == 0)
        printf("Time with leader = %lf\n", ans);

    free(buf);
    free(tmp);

    return 0;
}

int main(int argc, char *argv[])
{

    MPI_Init(&argc, &argv);
    int myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    // input argument test
    // if(myrank == 0)
    // {

    //     printf("=====================\n");
    //     printf("Px = %s\n", argv[1]);
    //     printf("N = %s\n", argv[2]);
    //     printf("num_time_steps = %s\n", argv[3]);
    //     printf("seed = %s\n", argv[4]);
    //     printf("stencil = %s\n", argv[5]);

    // }

    // The number of data points per processor
    int N = atoi(argv[2]);
    int px = atoi(argv[1]);
    int nt = atoi(argv[3]);
    int seed = atoi(argv[4]);

    // Calling the hierarchical implementation
    leader(px, N, nt, seed, myrank, size);

    // Barrier to ensure the leader call has ended on all processes
    // (Redundant in view of the last Reduce for printing maximum time)
    MPI_Barrier(MPI_COMM_WORLD);

    // Calling the normal implementation
    non_leader(px, N, nt, seed, myrank, size);

    MPI_Finalize();

    return 0;
}