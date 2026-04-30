# GitHub Practice Workspace



## Overview



This repository is a shared workspace for practicing version control using Git and GitHub. It is designed to help you become familiar with basic workflows such as pulling, adding files, committing changes, and collaborating with others.



The goal is not perfection, but practice.

---



## Task on 27.04


* Task3 in the Statistic folder



---



## Rules and Expectations



* There is **no strict format** for your folder

* You are free to experiment and make mistakes

* Mistakes are part of the learning process



This repository is a **practice environment**, not a graded submission.



---



## What You Should Do



* Create your own folder

* Update it throughout the semester

* Optionally collaborate with others

* Do not worry about mistakes 




---



## Note: branch



When working with a branch like `zaza`, you can think of Git commands as separating **creation** and **movement**. First, `git branch zaza` simply *creates* a new branch pointer called `zaza`, but you are still on your previous branch. To actually start working on it, you use `git switch zaza`, which moves your working directory onto that branch (the command `git checkout zaza` does the same). Once on `zaza`, any `git add` and `git commit` will belong to that branch. If you now want to publish it to the remote repository, you use `git push -u origin zaza`. The `-u` (or `--set-upstream`) is important the **first time**: it tells Git that your local `zaza` should be linked to `origin/zaza`. After this link is established, future commands like `git push` or `git pull` automatically know where to send to or retrieve from, so you do not need to specify the branch again.



---



## Disclaimer



This repository is intended **only for individual educational practice**.

Content may be incomplete, inconsistent, or contain errors.








