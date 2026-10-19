"""Streamlit entry point: python -m streamlit run app.py."""
import runpy

runpy.run_module("dental_fdi.ui.app", run_name="__main__")
