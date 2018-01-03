import subprocess
from glob import glob as glob
import os

#segment finistere
classif_str = "/home/tpostadjian/anaconda3/bin/python ../test/main.py -i ../test_set/finistere -d ../test/finistere -s t -r 0.2"
subprocess.call(classif_str, shell=True)

classif_str = "/home/tpostadjian/anaconda3/bin/python ../test/main.py -i ../test_set/finistere -d ../test/finistere -s t -r 0.4"
subprocess.call(classif_str, shell=True)

classif_str = "/home/tpostadjian/anaconda3/bin/python ../test/main.py -i ../test_set/finistere -d ../test/finistere -s t -r 0.6"
subprocess.call(classif_str, shell=True)

classif_str = "/home/tpostadjian/anaconda3/bin/python ../test/main.py -i ../test_set/finistere -d ../test/finistere -s t -r 0.8"
subprocess.call(classif_str, shell=True)

#pixel finistere
classif_str = "/home/tpostadjian/anaconda3/bin/python ../test/main.py -i ../test_set/finistere -d ../test/finistere -s f -r 0.6"
subprocess.call(classif_str, shell=True)
#####
#####
#segment gironde
classif_str = "/home/tpostadjian/anaconda3/bin/python ../test/main.py -i ../test_set/gironde -d ../test/gironde -s t -r 0.2"
subprocess.call(classif_str, shell=True)

classif_str = "/home/tpostadjian/anaconda3/bin/python ../test/main.py -i ../test_set/gironde -d ../test/gironde -s t -r 0.4"
subprocess.call(classif_str, shell=True)

classif_str = "/home/tpostadjian/anaconda3/bin/python ../test/main.py -i ../test_set/gironde -d ../test/gironde -s t -r 0.6"
subprocess.call(classif_str, shell=True)

classif_str = "/home/tpostadjian/anaconda3/bin/python ../test/main.py -i ../test_set/gironde -d ../test/gironde -s t -r 0.8"
subprocess.call(classif_str, shell=True)

#pixel gironde
classif_str = "/home/tpostadjian/anaconda3/bin/python ../test/main.py -i ../test_set/finistere -d ../test/finistere -s f -r 0.6"
subprocess.call(classif_str, shell=True)


############### GIRONDE

list_img = glob("../test_set/gironde/*")
for tile in list_img:
    tile = os.path.basename(tile)
    valid_str = "/usr/bin/python2.7 validation/validation.py -l /media/tpostadjian/Data/These/Test/data/gironde/BDTOPO/A_RESEAU_ROUTIER/" \
                "route_primaire_secondaire_poly.shp /media/tpostadjian/Data/These/Test/data/gironde/BDTOPO/E_BATI/BATI_INDIFFERENCIE.SHP " \
                "/media/tpostadjian/Data/These/Test/data/gironde/BDTOPO/F_VEGETATION/ZONE_VEGETATION.SHP " \
                "/media/tpostadjian/Data/These/Test/data/gironde/BDTOPO/D_HYDROGRAPHIE/SURFACE_EAU.SHP " \
                "/media/tpostadjian/Data/These/Test/data/gironde/RPG_1-0_SHP_LAMB93_R075-2014/ILOTS_ANONYMES.shp " \
                "-r ../test/gironde/"+tile+"/"+tile+"_classif_20.tif -o . -f None -s f"
    subprocess.call(valid_str, shell=True)
    valid_str = "/usr/bin/python2.7 validation/validation.py -l /media/tpostadjian/Data/These/Test/data/gironde/BDTOPO/A_RESEAU_ROUTIER/" \
                "route_primaire_secondaire_poly.shp /media/tpostadjian/Data/These/Test/data/gironde/BDTOPO/E_BATI/BATI_INDIFFERENCIE.SHP " \
                "/media/tpostadjian/Data/These/Test/data/gironde/BDTOPO/F_VEGETATION/ZONE_VEGETATION.SHP " \
                "/media/tpostadjian/Data/These/Test/data/gironde/BDTOPO/D_HYDROGRAPHIE/SURFACE_EAU.SHP " \
                "/media/tpostadjian/Data/These/Test/data/gironde/RPG_1-0_SHP_LAMB93_R075-2014/ILOTS_ANONYMES.shp " \
                "-r ../test/gironde/"+tile+"/"+tile+"_classif_40.tif -o . -f None -s f"
    subprocess.call(valid_str, shell=True)
    valid_str = "/usr/bin/python2.7 validation/validation.py -l /media/tpostadjian/Data/These/Test/data/gironde/BDTOPO/A_RESEAU_ROUTIER/" \
                "route_primaire_secondaire_poly.shp /media/tpostadjian/Data/These/Test/data/gironde/BDTOPO/E_BATI/BATI_INDIFFERENCIE.SHP " \
                "/media/tpostadjian/Data/These/Test/data/gironde/BDTOPO/F_VEGETATION/ZONE_VEGETATION.SHP " \
                "/media/tpostadjian/Data/These/Test/data/gironde/BDTOPO/D_HYDROGRAPHIE/SURFACE_EAU.SHP " \
                "/media/tpostadjian/Data/These/Test/data/gironde/RPG_1-0_SHP_LAMB93_R075-2014/ILOTS_ANONYMES.shp " \
                "-r ../test/gironde/"+tile+"/"+tile+"_classif_60.tif -o . -f None -s f"
    subprocess.call(valid_str, shell=True)
    valid_str = "/usr/bin/python2.7 validation/validation.py -l /media/tpostadjian/Data/These/Test/data/gironde/BDTOPO/A_RESEAU_ROUTIER/" \
                "route_primaire_secondaire_poly.shp /media/tpostadjian/Data/These/Test/data/gironde/BDTOPO/E_BATI/BATI_INDIFFERENCIE.SHP " \
                "/media/tpostadjian/Data/These/Test/data/gironde/BDTOPO/F_VEGETATION/ZONE_VEGETATION.SHP " \
                "/media/tpostadjian/Data/These/Test/data/gironde/BDTOPO/D_HYDROGRAPHIE/SURFACE_EAU.SHP " \
                "/media/tpostadjian/Data/These/Test/data/gironde/RPG_1-0_SHP_LAMB93_R075-2014/ILOTS_ANONYMES.shp " \
                "-r ../test/gironde/"+tile+"/"+tile+"_classif_80.tif -o . -f None -s f"
    subprocess.call(valid_str, shell=True)
    valid_str = "/usr/bin/python2.7 validation/validation.py -l /media/tpostadjian/Data/These/Test/data/gironde/BDTOPO/A_RESEAU_ROUTIER/" \
                "route_primaire_secondaire_poly.shp /media/tpostadjian/Data/These/Test/data/gironde/BDTOPO/E_BATI/BATI_INDIFFERENCIE.SHP " \
                "/media/tpostadjian/Data/These/Test/data/gironde/BDTOPO/F_VEGETATION/ZONE_VEGETATION.SHP " \
                "/media/tpostadjian/Data/These/Test/data/gironde/BDTOPO/D_HYDROGRAPHIE/SURFACE_EAU.SHP " \
                "/media/tpostadjian/Data/These/Test/data/gironde/RPG_1-0_SHP_LAMB93_R075-2014/ILOTS_ANONYMES.shp " \
                "-r ../test/gironde/"+tile+"/"+tile+"_classif_pix.tif -o . -f None -s f"
    subprocess.call(valid_str, shell=True)


############### FINISTERE

list_img = glob("../test_set/finistere/*")
for tile in list_img:
    tile = os.path.basename(tile)
    valid_str = "/usr/bin/python2.7 validation/validation.py -l /media/tpostadjian/Data/These/Test/data/brest/Database/BDTOPO/BDTOPO/1_DONNEES_LIVRAISON_2015-04-00253/BDT_2-1_SHP_LAMB93_D029-ED151/A_RESEAU_ROUTIER" \
                "route_primaire_secondaire_buffer.shp /media/tpostadjian/Data/These/Test/data/brest/Database/BDTOPO/BDTOPO/1_DONNEES_LIVRAISON_2015-04-00253/BDT_2-1_SHP_LAMB93_D029-ED151/E_BATI/BATI_INDIFFERENCIE.SHP " \
                "/media/tpostadjian/Data/These/Test/data/brest/Database/BDFORET/BDFORET_2-0/1_DONNEES_LIVRAISON/BDF_2_SHP_LAMB93_29/FORMATION_VEGETALE.shp " \
                "/media/tpostadjian/Data/These/Test/data/brest/Database/BDTOPO/BDTOPO/1_DONNEES_LIVRAISON_2015-04-00253/BDT_2-1_SHP_LAMB93_D029-ED151/D_HYDROGRAPHIE/SURFACE_EAU.SHP " \
                "/media/tpostadjian/Data/These/Test/data/brest/Database/RPG/RPG_2012_029.shp " \
                "-r ../test/finistere/"+tile+"/"+tile+"_classif_20.tif -o . -f None -s f"
    subprocess.call(valid_str, shell=True)

    valid_str = "/usr/bin/python2.7 validation/validation.py -l /media/tpostadjian/Data/These/Test/data/brest/Database/BDTOPO/BDTOPO/1_DONNEES_LIVRAISON_2015-04-00253/BDT_2-1_SHP_LAMB93_D029-ED151/A_RESEAU_ROUTIER" \
                    "route_primaire_secondaire_buffer.shp /media/tpostadjian/Data/These/Test/data/brest/Database/BDTOPO/BDTOPO/1_DONNEES_LIVRAISON_2015-04-00253/BDT_2-1_SHP_LAMB93_D029-ED151/E_BATI/BATI_INDIFFERENCIE.SHP " \
                    "/media/tpostadjian/Data/These/Test/data/brest/Database/BDFORET/BDFORET_2-0/1_DONNEES_LIVRAISON/BDF_2_SHP_LAMB93_29/FORMATION_VEGETALE.shp " \
                    "/media/tpostadjian/Data/These/Test/data/brest/Database/BDTOPO/BDTOPO/1_DONNEES_LIVRAISON_2015-04-00253/BDT_2-1_SHP_LAMB93_D029-ED151/D_HYDROGRAPHIE/SURFACE_EAU.SHP " \
                    "/media/tpostadjian/Data/These/Test/data/brest/Database/RPG/RPG_2012_029.shp " \
                    "-r ../test/finistere/"+tile+"/"+tile+"_classif_40.tif -o . -f None -s f"
    subprocess.call(valid_str, shell=True)

    valid_str = "/usr/bin/python2.7 validation/validation.py -l /media/tpostadjian/Data/These/Test/data/brest/Database/BDTOPO/BDTOPO/1_DONNEES_LIVRAISON_2015-04-00253/BDT_2-1_SHP_LAMB93_D029-ED151/A_RESEAU_ROUTIER" \
                    "route_primaire_secondaire_buffer.shp /media/tpostadjian/Data/These/Test/data/brest/Database/BDTOPO/BDTOPO/1_DONNEES_LIVRAISON_2015-04-00253/BDT_2-1_SHP_LAMB93_D029-ED151/E_BATI/BATI_INDIFFERENCIE.SHP " \
                    "/media/tpostadjian/Data/These/Test/data/brest/Database/BDFORET/BDFORET_2-0/1_DONNEES_LIVRAISON/BDF_2_SHP_LAMB93_29/FORMATION_VEGETALE.shp " \
                    "/media/tpostadjian/Data/These/Test/data/brest/Database/BDTOPO/BDTOPO/1_DONNEES_LIVRAISON_2015-04-00253/BDT_2-1_SHP_LAMB93_D029-ED151/D_HYDROGRAPHIE/SURFACE_EAU.SHP " \
                    "/media/tpostadjian/Data/These/Test/data/brest/Database/RPG/RPG_2012_029.shp " \
                    "-r ../test/finistere/"+tile+"/"+tile+"_classif_60.tif -o . -f None -s f"
    subprocess.call(valid_str, shell=True)

    valid_str = "/usr/bin/python2.7 validation/validation.py -l /media/tpostadjian/Data/These/Test/data/brest/Database/BDTOPO/BDTOPO/1_DONNEES_LIVRAISON_2015-04-00253/BDT_2-1_SHP_LAMB93_D029-ED151/A_RESEAU_ROUTIER" \
                    "route_primaire_secondaire_buffer.shp /media/tpostadjian/Data/These/Test/data/brest/Database/BDTOPO/BDTOPO/1_DONNEES_LIVRAISON_2015-04-00253/BDT_2-1_SHP_LAMB93_D029-ED151/E_BATI/BATI_INDIFFERENCIE.SHP " \
                    "/media/tpostadjian/Data/These/Test/data/brest/Database/BDFORET/BDFORET_2-0/1_DONNEES_LIVRAISON/BDF_2_SHP_LAMB93_29/FORMATION_VEGETALE.shp " \
                    "/media/tpostadjian/Data/These/Test/data/brest/Database/BDTOPO/BDTOPO/1_DONNEES_LIVRAISON_2015-04-00253/BDT_2-1_SHP_LAMB93_D029-ED151/D_HYDROGRAPHIE/SURFACE_EAU.SHP " \
                    "/media/tpostadjian/Data/These/Test/data/brest/Database/RPG/RPG_2012_029.shp " \
                    "-r ../test/finistere/"+tile+"/"+tile+"_classif_80.tif -o . -f None -s f"
    subprocess.call(valid_str, shell=True)

    valid_str = "/usr/bin/python2.7 validation/validation.py -l /media/tpostadjian/Data/These/Test/data/brest/Database/BDTOPO/BDTOPO/1_DONNEES_LIVRAISON_2015-04-00253/BDT_2-1_SHP_LAMB93_D029-ED151/A_RESEAU_ROUTIER" \
                    "route_primaire_secondaire_buffer.shp /media/tpostadjian/Data/These/Test/data/brest/Database/BDTOPO/BDTOPO/1_DONNEES_LIVRAISON_2015-04-00253/BDT_2-1_SHP_LAMB93_D029-ED151/E_BATI/BATI_INDIFFERENCIE.SHP " \
                    "/media/tpostadjian/Data/These/Test/data/brest/Database/BDFORET/BDFORET_2-0/1_DONNEES_LIVRAISON/BDF_2_SHP_LAMB93_29/FORMATION_VEGETALE.shp " \
                    "/media/tpostadjian/Data/These/Test/data/brest/Database/BDTOPO/BDTOPO/1_DONNEES_LIVRAISON_2015-04-00253/BDT_2-1_SHP_LAMB93_D029-ED151/D_HYDROGRAPHIE/SURFACE_EAU.SHP " \
                    "/media/tpostadjian/Data/These/Test/data/brest/Database/RPG/RPG_2012_029.shp " \
                    "-r ../test/finistere/"+tile+"/"+tile+"_classif_pix.tif -o . -f None -s f"
    subprocess.call(valid_str, shell=True)


