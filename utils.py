import glob
import io
import base64
from time import sleep
import numpy as np
import pandas as pd
import gym
from gym.wrappers import Monitor
from typing import Callable
from IPython.display import HTML
from pyvirtualdisplay import Display
from IPython import display as ipythondisplay
from IPython.display import clear_output
import matplotlib.pyplot as plt


display = Display(visible=0, size=(1400, 900))
display.start()


def show_video():
    mp4list = glob.glob("video/*.mp4")
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, "r+b").read()
        encoded = base64.b64encode(video)
        ipythondisplay.display(
            HTML(
                data="""<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>""".format(
                    encoded.decode("ascii")
                )
            )
        )
    else:
        print("Could not find video")


def wrap_env(env):

    env = Monitor(env, "./video", force=True)
    return env
