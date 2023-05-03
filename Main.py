import sys

from dagonstar.dagon import Workflow
from dagonstar.dagon.task import DagonTask, TaskType

if __name__ == '__main__':

    if len(sys.argv) != 5:
        print("Usage: python " + str(sys.argv[0]) + "source_filename mask_filename destination_filename border_filename")
        sys.exit(-1)

    src_filename = sys.argv[1]
    mask_filename = sys.argv[2]
    destination_filename = sys.argv[3]
    border_filename = sys.argv[4]

    workflow = Workflow("DataFlow-Test")

    # task create file
    taskA = DagonTask(TaskType.BATCH, "A", "Python CreateFile.py " + mask_filename + " " + destination_filename)
    workflow.add_task(taskA)

    for i in range(9):

        taskB = DagonTask(TaskType.BATCH, "B", "Python Temperature.py " + src_filename + "_tem.nc" + " " +
                          mask_filename + " " + destination_filename + " " + border_filename + " " + str(i))

        taskC = DagonTask(TaskType.BATCH, "C", "Python Salinity.py " + src_filename + "_sal.nc" + " " + mask_filename
                          + " " + destination_filename + " " + border_filename + " " + str(i))

        taskD = DagonTask(TaskType.BATCH, "D", "Python SeaSurfaceHeight.py " + src_filename + "_ssh.nc" + " " +
                          mask_filename + " " + destination_filename + " " + border_filename + " " + str(i))

        taskE = DagonTask(TaskType.BATCH, "E", "Python Current.py " + src_filename + "_cur.nc" + " " +
                          mask_filename + " " + destination_filename + " " + border_filename + " " + str(i))

        workflow.add_task(taskB)
        workflow.add_task(taskC)
        workflow.add_task(taskD)
        workflow.add_task(taskE)

    workflow.make_dependencies()

    workflow.run()


