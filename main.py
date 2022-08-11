import traceback
import sys
from collections import OrderedDict
import json
import logging
from decimal import Decimal
from enum import Enum
import os
import toga
from run import run
from config import Config
from utils import is_standalone


def exception_to_str(e: Exception):
    return "".join(traceback.format_exception(sys.exc_info()[0], sys.exc_info()[1], sys.exc_info()[2]))


def var_to_label(var: str):
    return " ".join(w[0].upper() + w[1:] for w in var.lower().split("_"))


def build_config_inputs(config, container, on_config_change):
    inputs = []
    for var in dir(config):
        if var.startswith("_"):
            continue
        value = getattr(config, var)
        value_type = type(value)
        box = toga.Box()
        label = toga.Label(var_to_label(var))
        label.style.flex = 1
        box.add(label)
        
        if value_type is float:
            input = toga.NumberInput(default=value, step=Decimal("0.01"), on_change=lambda input, var=var: on_config_change(var, float(input.value)))
        if value_type is int:
            input = toga.NumberInput(default=value, step=1, on_change=lambda input, var=var: on_config_change(var, int(input.value)))
        elif value_type is bool:
            input = toga.Switch("", is_on=value, on_toggle=lambda input, var=var: on_config_change(var, input.is_on))
        elif issubclass(value_type, Enum):
            input = toga.Selection(items=[e for e in dir(type(value)) if not e.startswith("_")], on_select=lambda input, var=var, default_type=value_type: on_config_change(var, getattr(default_type, input.value)))
            input.value = value.name
        
        inputs.append(input)
        input_box = toga.Box()
        input_box.add(input)
        input_box.style.flex = 0
        input_box.style.padding_left = input_box.style.padding_right = 75
        box.add(input_box)
        box.style.padding_left = box.style.padding_right = 15
        box.style.padding_top = box.style.padding_bottom = 2
        box.style.flex = 1
        container.add(box)
    return inputs


def build(app: toga.App):
    sleep_duration = 0.01
    executable_path = os.path.realpath(sys.executable) if is_standalone() else os.path.realpath(__file__)
    config_path = os.path.join(os.path.dirname(executable_path), os.path.splitext(os.path.basename(executable_path))[0] + "_config.json")
    log_path = os.path.join(os.path.dirname(executable_path), os.path.splitext(os.path.basename(executable_path))[0] + "_log.txt")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

    def persist_config(config, config_path):
        obj = OrderedDict()
        for var in dir(config):
            if var.startswith("_"):
                continue
            value = getattr(config, var)
            if issubclass(type(value), Enum):
                value = value.name
            obj[var] = value
        try:
            with open(config_path, "w") as f:
                json.dump(obj, f)
        except Exception as e:
            logging.info(f"Unable to persist config: {str(e)}")

    def load_config(config_path):
        try:
            with open(config_path) as f:
                obj = json.load(f)
            for k, v in obj.copy().items():
                if not hasattr(Config, k):
                    continue
                if issubclass(type(getattr(Config, k)), Enum):
                    obj[k] = getattr(type(getattr(Config, k)), v)
            config = Config(**obj)
        except Exception as e:
            logging.info(f"Unable to load config: {str(e)}")
            config = Config()
        return config

    config = load_config(config_path)

    main_box = toga.Box()
    main_box.style.direction = "column"

    data_dir_box = toga.Box()
    data_dir_box.style.padding = 15
    data_dir_label = toga.TextInput(readonly=True, initial=config.data_dir)
    data_dir_label.style.flex = 1
    data_dir_label.style.padding_left = 15

    def update_data_dir(_):
        def on_result(_, data_dir):
            if data_dir is not None:
                data_dir_label.value = str(data_dir)
                config.data_dir = str(data_dir)
                persist_config(config, config_path)

        app.main_window.select_folder_dialog("Select Data Directory", on_result=on_result)

    update_data_dir_button = toga.Button("Select Data Directory", on_press=update_data_dir)
    data_dir_box.add(update_data_dir_button)
    data_dir_box.add(data_dir_label)
    main_box.add(data_dir_box)

    def enable_inputs(inputs, enable):
        for input in inputs:
            input.enabled = enable

    def on_config_change(var, value):
        setattr(config, var, value)
        persist_config(config, config_path)

    inputs = build_config_inputs(config, main_box, on_config_change) + [update_data_dir_button, data_dir_label]

    status_box = toga.Box()
    status_box.style.padding = 15

    terminate_run = False
    last_status_update = None
    def run_wrapper(*args, **kwargs):
        nonlocal terminate_run, last_status_update
        try:
            yield sleep_duration
            for status_update in run(*args, **kwargs):
                if terminate_run:
                    break
                if status_update is not None:
                    progressbar.value = status_update.current_transect_idx / status_update.total_transects
                    last_status_update = status_update
                yield sleep_duration
            if not terminate_run:
                progressbar.value = 1
        except Exception as e:
            exception_str = exception_to_str(e)
            logging.error(exception_str)
            status_str = ""
            if last_status_update is not None:
                status_str += f" at transect '{last_status_update.current_transect_id}'"
                status_str += f" and detection '{last_status_update.current_detection_id}'" if last_status_update.current_detection_id is not None else ""
            app.main_window.info_dialog("Error", f"An error occured{status_str}: {str(e)}\nDetails:\n{exception_str}")
        finally:
            run_button.label = "Start"
            enable_inputs(inputs, True)
            terminate_run = False
            last_status_update = None

    def on_run(_):
        nonlocal terminate_run
        if run_button.label == "Start":
            run_button.label = "Stop"
            enable_inputs(inputs, False)
            app.add_background_task(lambda _: run_wrapper(config))
        elif run_button.label == "Stop":
            terminate_run = True
            run_button.label = "Start"
            progressbar.value = 0
            enable_inputs(inputs, True)

    run_button = toga.Button("Start", on_press=on_run)
    progressbar_box = toga.Box()
    progressbar_box.style.flex = 1
    progressbar_box.style.direction = "row"
    progressbar_box.style.alignment = "center"
    progressbar = toga.ProgressBar()
    progressbar.style.flex = 1
    progressbar.style.padding_left = 15
    progressbar_box.add(progressbar)

    status_box.add(run_button)
    status_box.add(progressbar_box)

    main_box.add(status_box)

    return main_box


def main():
    return toga.App('Distance Estimation', 'xyz.haucke.distance_sampling', startup=build)


if __name__ == '__main__':
    main().main_loop()