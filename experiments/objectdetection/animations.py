#!/usr/bin/env python3
#
# animations.py

from halo import Halo
import time

# All available spinner types in Halo
spinners = [
    'dots', 'dots2', 'dots3', 'dots4', 'dots5', 'dots6', 'dots7', 'dots8',
    'dots9', 'dots10', 'dots11', 'dots12', 'line', 'line2', 'pipe', 'simpleDots',
    'simpleDotsScrolling', 'star', 'star2', 'flip', 'hamburger', 'growVertical',
    'growHorizontal', 'balloon', 'balloon2', 'noise', 'bounce', 'boxBounce',
    'boxBounce2', 'triangle', 'arc', 'circle', 'squareCorners', 'circleQuarters',
    'circleHalves', 'squish', 'toggle', 'toggle2', 'toggle3', 'toggle4', 'toggle5',
    'toggle6', 'toggle7', 'toggle8', 'arrow', 'arrow2', 'arrow3', 'bouncingBar',
    'bouncingBall', 'smiley', 'monkey', 'hearts', 'clock', 'earth', 'moon', 'runner',
    'pong', 'shark', 'dqpb'
]

def demo_spinners():
    for spinner_name in spinners:
        spinner = Halo(text=f'Spinner: {spinner_name}', spinner=spinner_name)
        spinner.start()
        time.sleep(1.5)  # Show each spinner for 1.5 seconds
        spinner.stop_and_persist(symbol='✔', text=f'Completed {spinner_name}')
        time.sleep(0.5)  # Brief pause before the next spinner

if __name__ == "__main__":
    demo_spinners()

