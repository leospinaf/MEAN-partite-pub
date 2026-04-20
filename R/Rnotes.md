# Notes

## Installation

Create empty directory

Open powershell / command prompt in directory
Create venv:
`python -m venv venv`

If using powershell - allow unsigned activate script to be run:

`Set-ExecutionPolicy Unrestricted -Scope CurrentUser`

Activate venv:
Powershell: `.\venv\Scripts\activate.ps1`
Comand prompt: `.\venv\Scripts\activate.bat`

On mac

`source venv/bin/activate`

(verify with `pip -V`)

Install moo from github:
First generate a [Personal Access Token](https://github.com/settings/tokens) with repo scope
(we only need to do this because the repo is currently private)

Install moo from repo:
`pip install git+https://{token}@github.com/UoMResearchIT/mo-community-detection-bipartite.git@package#egg=moo`
(replacing `{token}` with the personal access token you just made (`ghp_....`))

The package and its dependencies will then be installed into the virtual environment

In Rstudio:
Create a new project, use existing directory, and select the directory you created (this will contain the `venv/` subdirectory)

`Rdemo.R` contains a brief example of useage

## Parallel processing

This works fine on WSL and, I suspect Linux/macOS (since they can all fork processes).

I couldn't get it to work on Windows - tried the suggested fixes at
https://github.com/rstudio/reticulate/issues/517
these didn't work


## Passing parameters

Pass integers as, e.g. `0L` and None as `py_none()`





