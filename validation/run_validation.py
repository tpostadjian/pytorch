import subprocess
from glob import glob as glob

list_img = glob("../test_set/finistere/*.tif")


classif_str = "test/main.py -i test_set/finistere -d test/finistere -s t -r 0.2"
subprocess.call(classif_str, shell=True)

valid_str = "/usr/bin/python2.7 validation/validation.py -l /media/tpostadjian/Data/These/Test/data/gironde/BDTOPO/A_RESEAU_ROUTIER/" \
            "route_primaire_secondaire_poly.shp /media/tpostadjian/Data/These/Test/data/gironde/BDTOPO/E_BATI/BATI_INDIFFERENCIE.SHP " \
            "/media/tpostadjian/Data/These/Test/data/gironde/BDTOPO/F_VEGETATION/ZONE_VEGETATION.SHP " \
            "/media/tpostadjian/Data/These/Test/data/gironde/BDTOPO/D_HYDROGRAPHIE/SURFACE_EAU.SHP " \
            "/media/tpostadjian/Data/These/Test/data/gironde/RPG_1-0_SHP_LAMB93_R075-2014/ILOTS_ANONYMES.shp " \
            "-r ../test/gironde/tile_26500_30500/tile_26500_30500_classif_20.tif -o . -f None -s f"

#subprocess.call(valid_str, shell=True)