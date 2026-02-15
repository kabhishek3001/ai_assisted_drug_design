# PyMOL Visualization Script for ADRB2 Drug Discovery
# Generated automatically

# Clean slate
reinitialize

# Load ADRB2 protein structure
load /home/abhishek30/adrb2_discovery/visualizations/structures/adrb2_2RH1.pdb, adrb2_protein

# Style the protein
hide everything, adrb2_protein
show cartoon, adrb2_protein
color marine, adrb2_protein
set cartoon_fancy_helices, 1

# Show binding site residues as sticks
select binding_site, resi 113+116+117+120+193+194+197+200+201+204+207+208+296+299+300+305+309+312+313
show sticks, binding_site
color yellow, binding_site
set stick_radius, 0.15, binding_site

# Load top candidate molecules

# Candidate 1: DL_03731 (pIC50=9.16)
load /home/abhishek30/adrb2_discovery/visualizations/structures/DL_03731.sdf, DL_03731
hide everything, DL_03731
show sticks, DL_03731
color red, DL_03731
set stick_radius, 0.2, DL_03731

# Candidate 2: DL_09145 (pIC50=8.63)
load /home/abhishek30/adrb2_discovery/visualizations/structures/DL_09145.sdf, DL_09145
hide everything, DL_09145
show sticks, DL_09145
color green, DL_09145
set stick_radius, 0.2, DL_09145

# Candidate 3: DL_03898 (pIC50=8.31)
load /home/abhishek30/adrb2_discovery/visualizations/structures/DL_03898.sdf, DL_03898
hide everything, DL_03898
show sticks, DL_03898
color cyan, DL_03898
set stick_radius, 0.2, DL_03898

# Candidate 4: DL_07779 (pIC50=8.26)
load /home/abhishek30/adrb2_discovery/visualizations/structures/DL_07779.sdf, DL_07779
hide everything, DL_07779
show sticks, DL_07779
color magenta, DL_07779
set stick_radius, 0.2, DL_07779

# Candidate 5: DL_01649 (pIC50=8.19)
load /home/abhishek30/adrb2_discovery/visualizations/structures/DL_01649.sdf, DL_01649
hide everything, DL_01649
show sticks, DL_01649
color orange, DL_01649
set stick_radius, 0.2, DL_01649

# Candidate 6: DL_07878 (pIC50=8.11)
load /home/abhishek30/adrb2_discovery/visualizations/structures/DL_07878.sdf, DL_07878
hide everything, DL_07878
show sticks, DL_07878
color purple, DL_07878
set stick_radius, 0.2, DL_07878

# Candidate 7: DL_04376 (pIC50=7.84)
load /home/abhishek30/adrb2_discovery/visualizations/structures/DL_04376.sdf, DL_04376
hide everything, DL_04376
show sticks, DL_04376
color pink, DL_04376
set stick_radius, 0.2, DL_04376

# Candidate 8: DL_09791 (pIC50=7.82)
load /home/abhishek30/adrb2_discovery/visualizations/structures/DL_09791.sdf, DL_09791
hide everything, DL_09791
show sticks, DL_09791
color lime, DL_09791
set stick_radius, 0.2, DL_09791

# Candidate 9: DL_05103 (pIC50=7.82)
load /home/abhishek30/adrb2_discovery/visualizations/structures/DL_05103.sdf, DL_05103
hide everything, DL_05103
show sticks, DL_05103
color salmon, DL_05103
set stick_radius, 0.2, DL_05103

# Candidate 10: DL_09450 (pIC50=7.81)
load /home/abhishek30/adrb2_discovery/visualizations/structures/DL_09450.sdf, DL_09450
hide everything, DL_09450
show sticks, DL_09450
color lightblue, DL_09450
set stick_radius, 0.2, DL_09450

# Center view on binding site
center binding_site
zoom binding_site, 8

# Set up nice rendering
set ray_shadows, 0
set antialias, 2
set line_smooth, 1
set depth_cue, 1
bg_color white

# Create different views
set_view (\
     0.999,    0.000,    0.000,\
     0.000,    0.999,    0.000,\
     0.000,    0.000,    0.999,\
     0.000,    0.000, -150.000,\
     0.000,    0.000,    0.000,\
    50.000,  250.000,    0.000 )

# Instructions
print ""
print "="*70
print "PyMOL Visualization Loaded!"
print "="*70
print ""
print "Commands:"
print "  - Click and drag to rotate"
print "  - Shift + drag to zoom"
print "  - Use mouse wheel to zoom in/out"
print ""
print "Objects:"
print "  - adrb2_protein: The ADRB2 receptor"
print "  - binding_site: Key binding site residues (yellow)"
print "  - DL_03731: Candidate 1 (pIC50=9.16, red)"
print "  - DL_09145: Candidate 2 (pIC50=8.63, green)"
print "  - DL_03898: Candidate 3 (pIC50=8.31, cyan)"
print "  - DL_07779: Candidate 4 (pIC50=8.26, magenta)"
print "  - DL_01649: Candidate 5 (pIC50=8.19, orange)"
print "  - DL_07878: Candidate 6 (pIC50=8.11, purple)"
print "  - DL_04376: Candidate 7 (pIC50=7.84, pink)"
print "  - DL_09791: Candidate 8 (pIC50=7.82, lime)"
print "  - DL_05103: Candidate 9 (pIC50=7.82, salmon)"
print "  - DL_09450: Candidate 10 (pIC50=7.81, lightblue)"

print ""
print "Toggle visibility:"
print "  disable DL_03731  # Hide candidate"
print "  disable DL_09145  # Hide candidate"
print "  disable DL_03898  # Hide candidate"
print "  disable DL_07779  # Hide candidate"
print "  disable DL_01649  # Hide candidate"
print "  disable DL_07878  # Hide candidate"
print "  disable DL_04376  # Hide candidate"
print "  disable DL_09791  # Hide candidate"
print "  disable DL_05103  # Hide candidate"
print "  disable DL_09450  # Hide candidate"

print ""
print "Export images:"
print "  ray 1200, 900"
print "  png adrb2_view.png, dpi=300"
print ""
print "="*70
