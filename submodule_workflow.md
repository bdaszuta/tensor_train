# submodule_workflow

New libraries go in `libs` and are incorporated as submodules.

To this end (each submodule is assumed to be a `git` repo. with at least one commit):
```bash
# note that `./libs/mva_submodule` can be a remote url
git submodule add ./libs/mva_submodule libs/mva_submodule

# to add this to the top-level repo
git add .gitmodules path/to/submodule
git commit -m "Adding submodule."
```