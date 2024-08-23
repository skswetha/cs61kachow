#include <mpi.h>

#include "coordinator.h"

#define READY 0
#define NEW_TASK 1
#define TERMINATE -1

int main(int argc, char *argv[]) {
  if (argc < 2) {
    printf("Error: not enough arguments\n");
    printf("Usage: %s [path_to_task_list]\n", argv[0]);
    return -1;
  }

  // TODO: implement Open MPI coordinator
  int num_tasks;
  task_t **tasks;
  
  if (read_tasks(argv[1], &num_tasks, &tasks)) {
    printf("Error reading task list from %s\n", argv[1]);
    return -1;
  }

  // call init on argc and argv
  MPI_Init(&argc, &argv);

  int proc_ID, num_procs;

  // call comm size with address of num processes
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
  // call comm rank with address of rank
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_ID);
  
  // is rank is 0, call manager func
  if (proc_ID == 0) {
  // manager func
      int nextTask = 0;
      MPI_Status status;
      int32_t message;

      while (nextTask < num_tasks) {
      // receove message from any source 
          MPI_Recv(&message, 1, MPI_INT32_T, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
      // get source process using status
          int source_proc = status.MPI_SOURCE;
      // send nextTask as message
          message = nextTask;
          MPI_Send(&message, 1, MPI_INT32_T, source_proc, 0, MPI_COMM_WORLD);
      // nexttask ++
          nextTask++;
    }

    //loop through all procs
    for (int i = 0; i < num_procs -1; i++) {
        // receive message from any source
          MPI_Recv(&message, 1, MPI_INT32_T, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
        // get source process using status
          int source_proc = status.MPI_SOURCE;
        // send terminate as message to process
          message = TERMINATE;
          MPI_Send(&message, 1, MPI_INT32_T, source_proc, 0, MPI_COMM_WORLD);
    }
    
  }
    
  else {
    //worker func
    int32_t message;
    while(true) {
        // let manager node worker is ready
        message = READY;
        MPI_Send(&message, 1, MPI_INT32_T, 0, 0, MPI_COMM_WORLD);
        // receive 1 message + store in message
        MPI_Recv(&message, 1, MPI_INT32_T, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // if messgae is terminate, break loop
        if (message == TERMINATE) {
            break;
        }
        if (execute_task(tasks[message])) {
            printf("Task %d failed\n", message);
          return -1;
        }
        free(tasks[message]->path);
        free(tasks[message]);
  }
}
   free(tasks);
  
  // call mpi finalize
  MPI_Finalize();
}
