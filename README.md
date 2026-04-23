# GitHub Practice Workspace



## Overview



This repository is a shared workspace for practicing version control using Git and GitHub. It is designed to help you become familiar with basic workflows such as pulling, adding files, committing changes, and collaborating with others.



The goal is not perfection, but practice.



---



## What You Should Do



### 1. Clone (pull) the repository



Start by cloning the repository to your local machine:



```bash

git clone <repository-url>

```



---



### 2. Create your own file



* Add a new file with any name you like

&#x20; (e.g. `yourname.txt`, `exercise1.py`, `notes.md`)

* This file will be your personal workspace for the semester



---



### 3. Update your file regularly



* Add content related to your tasks, exercises, or notes

* Commit and push updates as you progress



Example workflow:



```bash

git add your_file_name

git commit -m "Update my file"

git push

```



---



### 4. Optional: Collaborate with others



* You may create a shared file with one or more classmates

* Work together by editing and updating the same file over time

* This helps you practice resolving merge conflicts and collaboration



---



## Rules and Expectations



* There is **no strict format** for your file

* You are free to experiment and make mistakes

* Mistakes are part of the learning process



This repository is a **practice environment**, not a graded submission.



---



## Notes



* Try to commit regularly rather than uploading everything at once

* Write short commit messages describing your changes

* Pull the latest version before starting new work:



```bash

git pull

```



---



## Notes: branch



When working with a branch like `zaza`, you can think of Git commands as separating **creation** and **movement**. First, `git branch zaza` simply *creates* a new branch pointer called `zaza`, but you are still on your previous branch. To actually start working on it, you use `git switch zaza`, which moves your working directory onto that branch (the command `git checkout zaza` does the same). Once on `zaza`, any `git add` and `git commit` will belong to that branch. If you now want to publish it to the remote repository, you use `git push -u origin zaza`. The `-u` (or `--set-upstream`) is important the **first time**: it tells Git that your local `zaza` should be linked to `origin/zaza`. After this link is established, future commands like `git push` or `git pull` automatically know where to send to or retrieve from, so you do not need to specify the branch again.



---



## Disclaimer



This repository is intended **only for individual educational practice**.

Content may be incomplete, inconsistent, or contain errors.



---



## Summary



* Create your own file

* Update it throughout the semester

* Optionally collaborate with others

* Do not worry about mistakes 





