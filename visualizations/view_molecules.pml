# Simple Molecule Viewer
# View top drug candidates

reinitialize


# DL_03731 (pIC50=9.16)
load /home/abhishek30/adrb2_discovery/visualizations/structures/DL_03731.sdf, DL_03731
show sticks, DL_03731
color red, DL_03731

# DL_09145 (pIC50=8.63)
load /home/abhishek30/adrb2_discovery/visualizations/structures/DL_09145.sdf, DL_09145
show sticks, DL_09145
color green, DL_09145

# DL_03898 (pIC50=8.31)
load /home/abhishek30/adrb2_discovery/visualizations/structures/DL_03898.sdf, DL_03898
show sticks, DL_03898
color cyan, DL_03898

# DL_07779 (pIC50=8.26)
load /home/abhishek30/adrb2_discovery/visualizations/structures/DL_07779.sdf, DL_07779
show sticks, DL_07779
color magenta, DL_07779

# DL_01649 (pIC50=8.19)
load /home/abhishek30/adrb2_discovery/visualizations/structures/DL_01649.sdf, DL_01649
show sticks, DL_01649
color orange, DL_01649

# DL_07878 (pIC50=8.11)
load /home/abhishek30/adrb2_discovery/visualizations/structures/DL_07878.sdf, DL_07878
show sticks, DL_07878
color purple, DL_07878

# DL_04376 (pIC50=7.84)
load /home/abhishek30/adrb2_discovery/visualizations/structures/DL_04376.sdf, DL_04376
show sticks, DL_04376
color pink, DL_04376

# DL_09791 (pIC50=7.82)
load /home/abhishek30/adrb2_discovery/visualizations/structures/DL_09791.sdf, DL_09791
show sticks, DL_09791
color lime, DL_09791

# DL_05103 (pIC50=7.82)
load /home/abhishek30/adrb2_discovery/visualizations/structures/DL_05103.sdf, DL_05103
show sticks, DL_05103
color red, DL_05103

# DL_09450 (pIC50=7.81)
load /home/abhishek30/adrb2_discovery/visualizations/structures/DL_09450.sdf, DL_09450
show sticks, DL_09450
color green, DL_09450

# Nice rendering
set stick_radius, 0.15
bg_color white
set antialias, 2
zoom

print "Loaded top drug candidates!"
print "Use 'disable molecule_name' to hide a molecule"
print "Use 'enable molecule_name' to show it again"
