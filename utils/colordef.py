import colorsys

darkpurple = (99/255, 106/255, 134/255)
pastelpurple = (199/255, 206/255, 234/255)

darkorange = (155/255, 118/255, 93/255)
pastelorange = (255/255, 218/255, 193/255)

darkblue = (92/255, 128/255, 141/255)
pastelblue = (192/255, 228/255, 241/255)

pastelred = (255/255, 154/255, 162/255)
darkred = (155/255, 54/255, 62/255)

pastelgreen = (225/255, 247/255, 208/255)
darkgreen = (145/255, 170/255, 126/255)

pastelyellow = (255/255, 255/255, 181/255)
darkyellow = (155/255, 155/255, 80/255)

def scale_lightness(rgb, scale_l):
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)

main_colors = [darkred, darkgreen, darkblue, darkorange, darkpurple, darkyellow]
secondary_colors = [pastelred, pastelgreen, pastelblue, pastelorange, pastelpurple, pastelyellow]