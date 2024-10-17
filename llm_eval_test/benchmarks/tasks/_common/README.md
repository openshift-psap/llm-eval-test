# Common files for Unitxt tasks

Symlink these files into each task that requires Unitxt.
Ensure that the symlinks are relative to the task directory.

```sh
mkdir new_task
cd new_task

ln -s ../_common/task.py ./
ln -s ../_common/unitxt ./
```
