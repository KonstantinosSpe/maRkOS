@echo off
rem Settings shared by the launchers in this folder. Called, not run on its own.
rem   THOR_DISTRO  the name of your WSL distribution (see: wsl -l)   default: Ubuntu
rem   THOR_REPO    this repository, found from where this file lives
if not defined THOR_DISTRO set "THOR_DISTRO=Ubuntu"
pushd "%~dp0..\.."
set "THOR_REPO=%CD%"
popd
