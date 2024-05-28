# [Libraries]{Reiforcement Learning}
import re
import time 
import gymnasium
import miniwob
from miniwob.action import ActionTypes

# [Registration of MiniWoB++ Environments]
gymnasium.register_envs(miniwob)

# [Environment]{click-button}
env = gymnasium.make('miniwob/click-button', render_mode='human')

# [try-finally block]{Proper  clean up}
try:
    # [Starting Episode]{Retrieving observation and info} 
    observation, info = env.reset()
    # [Utterance]
    utterance = observation['utterance']
    # [Regular Expression Pattern]
    pattern = r'"(.*?)"'
    # [Matches]{Finidng all matches of the pattern in utterance}
    matches = re.findall(pattern, utterance)
    # [Time interval]
    time.sleep(2)
    # [Looping Matches]
    for match in matches:
        #[Looping Through DOM Elements]
        for element in observation['dom_elements']:
            #[Breaking search]{If the element has it's text equal to the match text we break the loop}
            if element["text"] == match:
                break 
    #[Action]{Setting action to click element}
    action = env.unwrapped.create_action(ActionTypes.CLICK_ELEMENT, ref=element["ref"])
    #[Episode variables]{Retrieving all variables for the action in every step}
    observation, reward, terminated, truncated, info = env.step(action)
    # [Time interval]
    time.sleep(2)
    # [Reward Display]{Printing Reward}
    print(reward)
    # [Terminating Episode]
    assert terminated is True
    if terminated:
            observation, info = env.reset()
finally: 
    # [Closing environment]{When everything is done we close the environment}
    env.close()

