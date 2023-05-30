Workflows, Jobs, Tasks
=======================================================================

A workflow is a definition of work to be done.  It is a list of tasks and gates.
Gates open allow tasks to be eligible for running.

A job is created when a workflow has been triggered to run and there is data to run it on.
A job knows about the workflow it is running.
Jobs are saved persisently in the database.  

A job is entered into the database by ``dataface.set_bx_jobs([bx_job_dict])``, which is called from the ``BxJob.register`` method.

A launcher is responsible for actually running a task in a CPU process.
When the process running a task finishes, the launcher harvests the task and updates it state in the database.  This happens in ``bx_launchers.base.harvest()``.

A task has the opportunity to write a brief summary of its execution for display in a grid.
It does this by either calling ``ExecutionSummary().append_raw()``.
Please see ``test_job_a.py`` for an example.

Jupyter notebooks can accomplish the same thing by writing the file ``execution.summary`` in the current directory.

After the task is harvested, the execution summary is saved into a field in the task table in the database by ``bx_launchers.base.get_post_run_fields_after_run()``.
The job table is also updated with each task's execution summary.

