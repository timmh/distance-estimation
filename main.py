import sys
from argparse import ArgumentParser
from collections import OrderedDict
import json
import logging
from decimal import Decimal
from enum import Enum
import os
from copy import deepcopy
import multiprocessing
from tqdm import tqdm
import toga
import certifi
import ssl
import urllib.request


# Ensure urllib (and downstream HTTP libraries) trust the certifi CA bundle when running from a packaged app.
_CERTIFI_CAFILE = certifi.where()
os.environ.setdefault("SSL_CERT_FILE", _CERTIFI_CAFILE)
os.environ.setdefault("REQUESTS_CA_BUNDLE", _CERTIFI_CAFILE)
_ssl_context = ssl.create_default_context(cafile=_CERTIFI_CAFILE)
urllib.request.install_opener(
    urllib.request.build_opener(urllib.request.HTTPSHandler(context=_ssl_context))
)

from config import Config
from utils import is_standalone, exception_to_str, EnumActionLowerCase, dirs


def var_to_label(var: str):
    return " ".join(w[0].upper() + w[1:] for w in var.lower().split("_"))


def build_config_inputs(config, container, on_config_change):
    inputs = []
    for var in dir(config):
        if var.startswith("_") or var == "data_dir":
            continue
        value = getattr(config, var)
        value_type = type(value)

        if value_type is float:
            input = toga.NumberInput(value=value, step=Decimal("0.01"), on_change=lambda input, var=var: on_config_change(var, float(input.value)))
        elif value_type is int:
            input = toga.NumberInput(value=value, step=1, on_change=lambda input, var=var: on_config_change(var, int(input.value)))
        elif value_type is bool:
            input = toga.Switch("", value=value, on_change=lambda input, var=var: on_config_change(var, input.is_on))
        elif issubclass(value_type, Enum):
            input = toga.Selection(items=[e for e in dir(type(value)) if not e.startswith("_")], on_change=lambda input, var=var, default_type=value_type: on_config_change(var, getattr(default_type, input.value)))
            input.value = value.name
        else:
            continue

        box = toga.Box()
        label = toga.Label(var_to_label(var))
        label.style.flex = 1
        box.add(label)

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
    os.makedirs(dirs.user_config_dir, exist_ok=True)
    os.makedirs(dirs.user_log_dir, exist_ok=True)
    config_path = os.path.join(dirs.user_config_dir, "config.json")
    log_path = os.path.join(dirs.user_log_dir, "log.txt")

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

    config_box = toga.Box()
    config_box.style.direction = "column"

    data_dir_box = toga.Box()
    data_dir_box.style.padding = 15
    data_dir_label = toga.TextInput(readonly=True, value=config.data_dir)
    data_dir_label.style.flex = 1
    data_dir_label.style.padding_left = 15

    async def update_data_dir(_):
        data_dir = await app.main_window.select_folder_dialog(
            "Select Data Directory",
            # on_result=on_result,
            initial_directory=os.path.realpath(config.data_dir) if os.path.isdir(config.data_dir) and config.data_dir != "" else None,
        )
        if data_dir is not None:
            data_dir_label.value = str(data_dir)
            config.data_dir = str(data_dir)
            persist_config(config, config_path)

    update_data_dir_button = toga.Button("Select Data Directory", on_press=update_data_dir)
    data_dir_box.add(update_data_dir_button)
    data_dir_box.add(data_dir_label)
    config_box.add(data_dir_box)

    def enable_inputs(inputs, enable):
        for input in inputs:
            input.enabled = enable

    def on_config_change(var, value):
        setattr(config, var, value)
        persist_config(config, config_path)

    inputs = build_config_inputs(config, config_box, on_config_change) + [update_data_dir_button, data_dir_label]

    config_box.style.flex = 1
    config_box_container = toga.ScrollContainer(content=config_box)
    config_box_container.style.flex = 1
    main_box.add(config_box_container)

    status_box = toga.Box()
    status_box.style.padding = 15

    terminate_run = False
    last_status_update = None
    def run_wrapper(*args, **kwargs):
        from run import run
        nonlocal terminate_run, last_status_update
        try:
            yield sleep_duration
            for status_update in run(*args, **kwargs, gui=True):
                if terminate_run:
                    break
                if status_update is not None:
                    progressbar.value = int(progressbar.max * status_update.current_transect_idx / status_update.total_transects)
                    last_status_update = status_update
                yield sleep_duration
            if not terminate_run:
                progressbar.value = progressbar.max
        except Exception as e:
            exception_str = exception_to_str(e)
            logging.error(exception_str)
            status_str = ""
            if last_status_update is not None:
                status_str += f" at transect '{last_status_update.current_transect_id}'"
                status_str += f" and detection '{last_status_update.current_detection_id}'" if last_status_update.current_detection_id is not None else ""
            app.main_window.info_dialog("Error", f"An error occured{status_str}: {str(e)}\nDetails:\n{exception_str}")
        finally:
            run_button.text = "Start"
            enable_inputs(inputs, True)
            terminate_run = False
            last_status_update = None

    def on_run(_):
        nonlocal terminate_run
        if run_button.text == "Start":
            run_button.text = "Stop"
            enable_inputs(inputs, False)
            app.add_background_task(lambda _: run_wrapper(deepcopy(config)))
        elif run_button.text == "Stop":
            terminate_run = True
            run_button.text = "Start"
            progressbar.value = 0
            enable_inputs(inputs, True)

    run_button = toga.Button("Start", on_press=on_run)
    progressbar_box = toga.Box()
    progressbar_box.style.flex = 1
    progressbar_box.style.direction = "row"
    progressbar_box.style.alignment = "center"
    progressbar = toga.ProgressBar(max=int(1e6))
    progressbar.style.flex = 1
    progressbar.style.padding_left = 15
    progressbar_box.add(progressbar)

    status_box.add(run_button)
    status_box.add(progressbar_box)

    main_box.add(status_box)

    return main_box


def cli(args):
    from run import run
    args = {k: v for k, v in args.__dict__.items() if not k.startswith("_") and k != "cli"}
    config = Config(**args)
    last_total = None
    last_idx = 0
    with tqdm() as progressbar:
        for status_update in run(config, gui=False):
            if status_update is not None:
                if last_total != status_update.total_transects:
                    last_total = status_update.total_transects
                    progressbar.reset(status_update.total_transects)
                progressbar.update(status_update.current_transect_idx - last_idx)
                last_idx = status_update.current_transect_idx
        if last_total is not None:
            progressbar.update(last_total - last_idx)


def main():

    argparser = ArgumentParser()
    argparser.add_argument("--cli", action="store_true", help="Enables CLI operation and disables GUI")
    default_config = Config()
    for var in dir(Config):
        value_type = type(getattr(default_config, var))
        default_value = getattr(default_config, var)
        if var.startswith("_"):
            continue
        if issubclass(value_type, Enum):
            argparser.add_argument(f"--{var}", type=value_type, default=default_value, action=EnumActionLowerCase)
        elif value_type is bool:
            if default_value is True:
                argparser.add_argument(f"--no_{var}", dest=var, action="store_false")
            else:
                argparser.add_argument(f"--{var}", dest=var, action="store_true")
        else:
            argparser.add_argument(f"--{var}", type=value_type, default=default_value)
    args = argparser.parse_args()
    if args.cli:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        cli(args)
    else:
        # We are in GUI mode. If on Windows, hide the console window that opened.
        if sys.platform == "win32":
            import ctypes
            # SW_HIDE = 0
            ctypes.windll.user32.ShowWindow(ctypes.windll.kernel32.GetConsoleWindow(), 0)

        # workaround for sys.stdout and sys.stderr being None on Windows without attached console.
        # see https://pyinstaller.org/en/v6.10.0/common-issues-and-pitfalls.html#sys-stdin-sys-stdout-and-sys-stderr-in-noconsole-windowed-applications-windows-only
        if sys.stdout is None:
            sys.stdout = open(os.devnull, "w")
        if sys.stderr is None:
            sys.stderr = open(os.devnull, "w")

        if is_standalone():
            icon = os.path.join(sys._MEIPASS, "assets", "icon.png")
        else:
            icon = os.path.join("assets", "icon.png")
        toga.App(
            "Distance Estimation",
            "xyz.haucke.distance_estimation",
            icon=icon,
            home_page="https://timm.haucke.xyz/publications/distance-estimation-animal-abundance",
            description="An application for estimating distances to animals in camera trap footage",
            author="Timm Haucke",
            startup=build,
        ).main_loop()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    if multiprocessing.parent_process() is None:
        main()