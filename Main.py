import sys

from dagonstar.dagon import Workflow
from dagonstar.dagon.task import DagonTask, TaskType

if __name__ == '__main__':

    if len(sys.argv) != 4:
        print("Usage: python " + str(sys.argv[0]) + " source_filename grid_filename destination_filename")
        sys.exit(-1)

    src_filename = sys.argv[1]
    grid_filename = sys.argv[2]
    destination_filename = sys.argv[3]

    workflow = Workflow("DataFlow-Test")

    # task create file
    taskA = DagonTask(TaskType.BATCH, "A", "CreateFile.py " + grid_filename + " " + destination_filename)
    workflow.add_task(taskA)

    for i in range(9):

        taskB = DagonTask(TaskType.BATCH, "B", "Temperature.py " + src_filename + " " + grid_filename + " " +
                          destination_filename + " " + str(i))

        taskC = DagonTask(TaskType.BATCH, "C", "Salinity.py " + src_filename + " " + grid_filename + " " +
                          destination_filename + " " + str(i))

        taskD = DagonTask(TaskType.BATCH, "D", "SeaSurfaceHeight.py " + src_filename + " " + grid_filename + " " +
                          destination_filename + " " + str(i))

        taskE = DagonTask(TaskType.BATCH, "E", "Current.py " + src_filename + " " + grid_filename + " " +
                          destination_filename + " " + str(i))

        workflow.add_task(taskB)
        workflow.add_task(taskC)
        workflow.add_task(taskD)
        workflow.add_task(taskE)

    workflow.make_dependencies()

    workflow.run()


