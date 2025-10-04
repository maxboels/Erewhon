# EMMA: Scaling Mobile Manipulation via Egocentric Human Data

Scaling mobile manipulation imitation learning is
bottlenecked by expensive mobile robot teleoperation. We present
Egocentric Mobile MAnipulation (EMMA), an end-to-end framework training mobile manipulation policies from human mobile
manipulation data with static robot data, sidestepping mobile
teleoperation. To accomplish this, we co-train human full-body
motion data with static robot data. In our experiments across
three real-world tasks, EMMA demonstrates comparable performance to baselines trained on teleoperated mobile robot
data (Mobile ALOHA), achieving higher or equivalent task
performance in full task success. We find that EMMA is able
to generalize to new spatial configurations and scenes, and we
observe positive performance scaling as we increase the hours of
human data, opening new avenues for scalable robotic learning
in real-world environments. Details of this project can be found
at https://ego-moma.github.io/.