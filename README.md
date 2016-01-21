# Simultaneus Localization and Mapping for mobile robots with Ackerman steering geometry 

This repository constains experiments for applying SLAM and Path Planning for Ackerman steering automobiles, with two kinds of maps. <br>
Description of the contents in each folder:
<ul>
<li>slam_landmarks map: SLAM using EKF and Particles Filters for landmarks maps.</li>
<li>localization and mapping_grid maps: MCL (localization) and Occupancy Grid Mapping algorithms for grid maps depicting scenarios of the Mechanical Enginering Department building at UNI. For details in modeling and implementation, check the paper inside the folder, or <a href="http://ronaldahmed.github.io/papers/autonomos.pdf">here</a>.</li>
<li>path-planning: Hybrid A* path planning algorithm implementation for an Ackerman steering HPI Buggy mini car. For details about the algorithm, refer to the paper:<br>
Dolgov, D., Thrun, S., Montemerlo, M., & Diebel, J. (2010). Path planning for autonomous vehicles in unknown semi-structured environments. International Journal of Robotics Research, 29, 485–501.
</li>
</ul>