# todo 

# what to alter in the codes:

    1. Fixing the random seeds for specific duration:
        in this case, we shall set the duration as a month,
        aiming at keeping the consistency of pred_error in different tests

    !!! DO REMEMBER TO SYNC THE CHANGE TO THE COPY ON REMOTE SERVERS !!!

# tests to run:

    1. For visualizing the MPC profile:
        Duration: a month -MAY
        Pred model: GT and Heuristic  !!! Need to add the heuristic of pv and ev back
        Demand charge: 0 and 18 $/kW/month 
        !!! RUN THIS LOCALLY !!!

    2. For tarcking the trends of P_grid_max:
        Duration: a month -MAY 
            Pred model: GT and Heuristic  !!! Need to add the heuristic of pv and ev back **better**
            Pred model: GT and Heuristic  !!! Exclude the heuristic of pv and ev
            # which to pick up depends on the results
        Demand charge: 0 and 18 $/kW/month 
        !!! RUN THIS LOCALLY !!!        

    3. For revealing the value of information:
        Duration: a month - 12 month the whole year
        Pred model: GT and Artificial Noises (uniform, uniform_pos, uniform_neg)
        Noise scales: [ 0.001,0.005,
                        0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,
                        0.15,0.20,0.25,0.30,0.35,0.40 ]
        Demand charge: 18 $/kW/month 
        !!! RUN THIS ON REMOTE SERVER !!!        

    4. Sensitivity analysis:
        Duration: a month -MAY  # if the results are not satisfying, may change it to another month
        Pred model: GT and Heuristic  !!! Need to add the heuristic of pv and ev back **better**
        Demand charge: 0.6
        Method1:2d:
            Group one - bat capacity & Dc price:
                Bat capacity: [ 1h, 2h, 3h, 4h, 5h, 6h, 7h, 8h, 9h, 10h, 11h, 12h ]
                Dc price: [ 3, 6, 9, 12, 15, 18, 21, 24, 27, 30 ] $/kW/month 
                PV capacity: 0.5 *Average bld load 
            Group two - bat capacity & PV Capacity:
                Bat capacity: [ 1h, 2h, 3h, 4h, 5h, 6h, 7h, 8h, 9h, 10h, 11h, 12h ]
                Dc price: 18 $/kW/month 
                PV capacity: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]*Average bld load
            Group three - PV capacity & Dc price:
                Bat capacity: 6h,
                Dc price: [ 3, 6, 9, 12, 15, 18, 21, 24, 27, 30 ] $/kW/month 
                PV capacity: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]*Average bld load
        Method1:3d:
            Group one :
                Bat capacity: [ 1h, 2h, 3h, 4h, 5h, 6h, 7h, 8h, 9h, 10h, 11h, 12h ]
                Dc price: [ 3, 6, 9, 12, 15, 18, 21, 24, 27, 30 ] $/kW/month 
                PV capacity: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]*Average bld load
        !!! RUN THIS ON REMOTE SERVER !!!    

