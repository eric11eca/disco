from distiller.prompt.core import Task
from distiller.prompt.lib.anli import AdversarialNliTask

TASK_DICT = {
    "adversarial_nli": AdversarialNliTask,
}


def get_task_class(task_name: str):
    task_class = TASK_DICT[task_name]
    assert issubclass(task_class, Task)
    return task_class
