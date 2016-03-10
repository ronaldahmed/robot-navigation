Maps = ['Grid', 'Jelly', 'L']
Positions = [str(i+1) for i in range(7)]
PositionSets = ['0','1']
UseCorpora = [1,2,3] # [1,2] # 

Corpus1_Director_Men = ['EDA', 'KXP', 'WLH',]
Corpus2_Director_Men = ['JJL', 'JXF', 'MXM', 'MJB', 'PXL', 'QNL']
Corpus3_Director_Men = ['JTM', 'KAJ', 'KXK', 'MHH', 'RRE', 'WAB',]
Corpus1_Director_Wmn = ['EMWC', 'KLS', 'TJS',]
Corpus2_Director_Wmn = ['BKW', 'BLO', 'JNN', 'LEN', 'MXP', 'TXG']
Corpus3_Director_Wmn = ['ARL', 'JLM', 'JXL', 'LCT', 'SCD', 'SMA']
Corpus1_Follower_Men = ['BAY','BXZ','BWB','CMN','DPM','FXM','IXC','IXL','JIC','MDP','OXA','PJA',
                        'RGG','RSB','SAA','SBA','SXD']
Corpus2_Follower_Men = ['AJT', 'APH', 'BLR', 'BMS','DAK', 'EXC', 'FBS', 'FXV', 'JDR', 'JMB', 'JMK',
                        'JRV', 'LXL', 'MAM', 'MMM', 'MTL', 'MPF', 'PJP', 'SKG', 'SHG', 'SMD']
Corpus3a_Follower_Men = ['MKD', 'WXJ','MDH', 'KMK', 'YXW']
Corpus3_Follower_Men = ['AAM', 'ARB', 'CNH', 'DAB', 'DGK', 'GNW', 'JAA', 'JRS', 'LAS', 'OVO', 'RSF', 'SAE']+Corpus3a_Follower_Men
Corpus1_Follower_Wmn = ['AMG','BBI','BRR','BXW','CEG','JMS','JXM','NXR','RCP','RRV','RXM','SLS','TIK','VMI',]
Corpus2_Follower_Wmn = ['AAK', 'AET', 'ANW', 'BMB', 'CAR', 'CJC', 'CLS', 'CMP', 'DNT', 'EJH', 'ERM', 'HJL',
                        'INL', 'KEW', 'KMT', 'KNT', 'LTV', 'MEB', 'MEC', 'RJR', 'SSP', 'SSW']
Corpus3a_Follower_Wmn = ['CXR', 'DHC', 'EMQ', 'RMH', 'TXN',]
Corpus3_Follower_Wmn = ['HRO', 'JFM', 'JRG', 'JXH', 'VAM']+Corpus3a_Follower_Wmn

Low_Eff_Followers = ['CAS','DXL','KXA','KXF','SNN','CVT','LAN','HXL','SEF','GNC','AAY','MPO']
#Lab_Followers = ['BJS','JPR','MTM',]
Frgn_Lang_Followers = ['CEG','FXM','KXA','NXR']

Directors1 = Corpus1_Director_Men + Corpus1_Director_Wmn
Directors2 = Corpus2_Director_Men + Corpus2_Director_Wmn
Directors3 = Corpus3_Director_Men + Corpus3_Director_Wmn
Followers1 = Corpus1_Follower_Men + Corpus1_Follower_Wmn
Followers2 = Corpus2_Follower_Men + Corpus2_Follower_Wmn
Followers3 = Corpus3_Follower_Men + Corpus3_Follower_Wmn

Director_Men = []
Director_Wmn = []
Follower_Men = []
Follower_Wmn = []
if 1 in UseCorpora:
    Director_Men.extend(Corpus1_Director_Men)
    Director_Wmn.extend(Corpus1_Director_Wmn)
    Follower_Men.extend(Corpus1_Follower_Men)
    Follower_Wmn.extend(Corpus1_Follower_Wmn)
if 2 in UseCorpora:
    Director_Men.extend(Corpus2_Director_Men)
    Director_Wmn.extend(Corpus2_Director_Wmn)
    Follower_Men.extend(Corpus2_Follower_Men)
    Follower_Wmn.extend(Corpus2_Follower_Wmn)
if 3 in UseCorpora:
    Director_Men.extend(Corpus3_Director_Men)
    Director_Wmn.extend(Corpus3_Director_Wmn)
    Follower_Men.extend(Corpus3_Follower_Men)
    Follower_Wmn.extend(Corpus3_Follower_Wmn)

Followers = Follower_Men + Follower_Wmn
Directors = Director_Men + Director_Wmn
All_Subjects = Followers + Directors
#All_Followers = Followers+Lab_Followers+Low_Eff_Followers
Eng_Prim_Followers = [x for x in Followers if not x in Frgn_Lang_Followers]

SubjGroups = {
#    'Dir_1_Men' : Corpus1_Director_Men,
#    'Dir_2_Men' : Corpus2_Director_Men,
#    'Dir_1_Wmn' : Corpus1_Director_Wmn,
#    'Dir_2_Wmn' : Corpus2_Director_Wmn,
#    'Dir_All' : Directors,
    'Fol_1_Men' : Corpus1_Follower_Men,
    'Fol_2_Men' : Corpus2_Follower_Men,
    'Fol_3_Men' : Corpus3_Follower_Men,
    'Fol_1_Wmn' : Corpus1_Follower_Wmn,
    'Fol_2_Wmn' : Corpus2_Follower_Wmn,
    'Fol_3_Wmn' : Corpus3_Follower_Wmn,
    'Fol_All' : Followers,
    'Fol_Men' : Follower_Men,
    'Fol_Wmn' : Follower_Wmn,
    'Fol_1' : Followers1,
    'Fol_2' : Followers2,
    'Fol_3' : Followers3,
#    'Low_Eff_Followers' : Low_Eff_Followers,
#    'Lab_Followers' : Lab_Followers,
#    'All_Followers' : All_Followers,
    'Fol_Frgn_Lang' : Frgn_Lang_Followers,
    'Fol_Eng_Prim' : Eng_Prim_Followers,
    'All_Subjects' : All_Subjects,
}

# Time director achieved navigation competency, per environment.
NavCompetency = {
    ('AER', 'Grid') : 4042.111,
    ('ARL', 'Jelly') : 793.173,
    ('BKW', 'L') : 2159.185,
    ('BLO', 'Grid') : 1885.115,
    ('BRS', 'Jelly') : -4077.294,
    ('EDA', 'Grid') : 855.469,
    ('EDA', 'Jelly') : 690.360+521.004,
    ('EDA', 'L') : 645.645,
    ('EMWC', 'Grid') : 1581.649,
    ('EMWC', 'Jelly') : 4431.002,
    ('EMWC', 'L') : 2201.606,
    ('JAG', 'Grid') : -4597.922,
    ('JEC', 'L') : 2733.466,
    ('JJL', 'Grid') : 1582.375,
    ('JLM', 'Grid') : 3181.093,
    ('JNN', 'Jelly') : 2859.799,
    ('JTM', 'Jelly') : 3052.442,
    ('JXF', 'Grid') : 1413.994,
    ('JXL', 'Grid') : 2612.421,
    ('KAJ', 'Grid') : 1070.112,
    ('KLS', 'Grid') : 800.127,
    ('KLS', 'Jelly') : 993.414,
    ('KLS', 'L') : 1255.005,
    ('KMH', 'L') : -978.218,
    ('KXK', 'L') : 1177.885,
    ('KXP', 'Grid') : 659.281,
    ('KXP', 'Jelly') : 1957.133,
    ('KXP', 'L') : 584.922,
    ('LCT', 'L') : 2900.357,
    ('LEN', 'Grid') : 1386.891,
    ('MHH', 'Jelly') : 1494.383,
    ('MJB', 'Jelly') : 735.968,
    ('MPO', 'Jelly') : -3619.893,
    ('MXM', 'Jelly') : 2248.130,
    ('MXP', 'L') : 1000.0,
    ('NXP', 'L') : 786.547,
    ('PXL', 'L') : 1318.743,
    ('RRE', 'Jelly') : -4958.090,
    ('QNL', 'L') : 565.214,
    ('SCD', 'Jelly') : 2068.209,
    ('SMA', 'L') : 2694.981,
    ('SRM', 'Jelly') : -(1290.046+2228.083),
    ('TDM', 'Grid') : 1564.681,
    ('THC', 'L') : 1168.049,
    ('TJS', 'Grid') : 2205.044,
    ('TJS', 'Jelly') : 2464.327,
    ('TJS', 'L') : 1306.029,
    ('TMH', 'Grid') : 2212.671,
    ('TXG', 'Jelly') : 1458.577,
    ('VLW', 'L') : -(1229.975+2332.931),
    ('WAB', 'Grid'): 1534.407,
    ('WLH', 'Grid') : 2160.712,
    ('WLH', 'Jelly') : 7212.956+2490.033,
    ('WLH', 'L') : 1369.753,
    }
