from naive import jpg
from full import tif

tifs = ['church', 'emir', 'icon', 'italil', 'lastochikino', 'lugano', 'melons', 'self_portrait', 'siren', 'three_generations']
jpgs = ['cathedral', 'monastery', 'tobolsk']

for im in tifs:
    tif(im)

for im in jpgs:
    jpg(im)