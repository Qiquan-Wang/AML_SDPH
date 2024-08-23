######## UPDATE 06/06/2023:
# SS = P1 now
# include ST = P2 samples again

######## datasets

#all_datasets = [14,26,27,31,36,37,38,42,43,44,55,56,57,58,59,60,61,65,66,67,68,71,72,73,74,76,77,79,80]
#discarded 46 = 36

datasets = {}
datasets['CTRL'] = [26,38,59,79]
datasets['U937'] = [37,58,72,67,31,36,57]
datasets['HL60'] = [56,44,65]
datasets['P1'] = [68,66,73,77,55,27]
datasets['P2'] = [60,74,42]
datasets['MNC'] = [71,76,61,80]

list_levels = {'CTRL' : [0, 0, 0, 0],
'U937' : [ 1,  1 , 7 , 8, 10, 10 ,10],
'HL60' : [23, 25 ,25],
'P1' : [10, 40, 44, 51, 60, 76],
'P2' : [59, 88, 90],
'MNC' : [53, 67, 75, 86] }

datasets_of_interest = [26,38,59,79,
                        37,58,72,67,31,36,57,
                        56,44,65,
                        68,66,73,77,55,27,
                        60,74,42,
                        71,76,61,80]

phases_of_interest = [0,0,0,0,
         1,1,1,1,1,1,1,
         1,1,1,
         1,2,2,2,2,1,
         2,2,2,
         1,2,2,2]

phases_labels = ['O','O','O','O',
         'I','I','I','I','I','I','I',
         'I','I','I',
         'I','II','II','II','II','I',
         'II','II','II',
         'I','II','II','II']

levels_of_interest = [0, 0, 0, 0,
                      1,  1,  7,  8, 10, 10, 10,
                      23, 25, 25,
                      10, 40, 44, 51, 60, 76,
                      59, 88, 90,
                      53, 67, 75, 86]


injected_datasets_of_interest = ['CTRL','CTRL','CTRL','CTRL',
                                'U937','U937','U937','U937','U937','U937','U937',
                                'HL60','HL60','HL60',
                                'P1','P1','P1','P1','P1','P1',
                                 'P2','P2','P2',
                                'MNC','MNC','MNC','MNC']

list_injected = ['CTRL','U937','HL60','P1','P2','MNC']
list_quadrant_names = ['PH0 SW', 'PH0 NW', 'PH1 SW', 'PH1 NW', 'PH1 NE', 'PH2 NW', 'PH2 NE']

names_of_interest = ['26 CTRL at 0%',
 '38 CTRL at 0%',
 '59 CTRL at 0%',
 '79 CTRL at 0%',
 '37 U937 at 1%',
 '58 U937 at 1%',
 '72 U937 at 7%',
 '67 U937 at 8%',
 '31 U937 at 10%',
 '36 U937 at 10%',
 '57 U937 at 10%',
 '56 HL60 at 23%',
 '44 HL60 at 25%',
 '65 HL60 at 25%',
 '68 P1 at 10%',
 '66 P1 at 40%',
 '73 P1 at 44%',
 '77 P1 at 51%',
 '55 P1 at 60%',
 '27 P1 at 76%',
 '60 P2 at 59%',
 '74 P2 at 88%',
 '42 P2 at 90%',    
 '71 MNC at 53%',
 '76 MNC at 67%',
 '61 MNC at 75%',
 '80 MNC at 86%']

paper_names_of_interest = ['CTRL at 0% (1)',
 'CTRL at 0% (2)',
 'CTRL at 0% (3)',
 'CTRL at 0% (4)',
 'U937 at 1% (1)',
 'U937 at 1% (2)',
 'U937 at 7%',
 'U937 at 8%',
 'U937 at 10% (1)',
 'U937 at 10% (2)',
 'U937 at 10% (3)',
 'HL60 at 23%',
 'HL60 at 25% (1)',
 'HL60 at 25% (2)',
 'P1 at 10%',
 'P1 at 40%',
 'P1 at 44%',
 'P1 at 51%',
 'P1 at 60%',
 'P1 at 76%',
 'P2 at 59%',
 'P2 at 88%',
 'P2 at 90%',    
 'MNC at 53%',
 'MNC at 67%',
 'MNC at 75%',
 'MNC at 86%']

label_colours = []
for i in phases_of_interest:
    if i == 0:
        label_colours.append("green")
    elif i == 1:
        label_colours.append("orange")
    else:
        label_colours.append("red")

######## folders

PH_folder = ''
data_folder = ''
segmented_folder = ''
denoised_folder = ''
