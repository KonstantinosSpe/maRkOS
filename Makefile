.PHONY: help install test lint plot ros

help:
	@echo "make install   editable install with the test and lint tools (headless OpenCV)"
	@echo "make test      run the whole test suite"
	@echo "make lint      ruff"
	@echo "make plot      regenerate docs/images/reach-envelope.png (needs matplotlib)"
	@echo "make ros       build the ROS 2 packages into ~/markos_ws (WSL / Ubuntu with ROS 2)"

install:
	pip install -e ".[dev,headless]"

test:
	pytest

lint:
	ruff check .

plot:
	python scripts/plot_reach_envelope.py

ros:
	scripts/wsl/build_ros.sh
