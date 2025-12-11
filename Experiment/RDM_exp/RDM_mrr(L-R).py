# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 09:47:57 2025
@author: Siebe Everaerts

Code for presenting random dot motion stimuli in 2 directions, recording responses
and giving feedback on the decision. Saves reaction time, response, accuracy, and
participant characteristics.

Use by pressing the "q" (or "a") key when dots are moving left and the "e" key when dots are moving right

"""

# Set up experiment -----------------------------------------------------------
# Import modules
import random
import numpy as np
from psychopy import visual as vis
from psychopy import event, core, data, gui
from scipy.stats import truncnorm, norm
import questplus as qp
import math
import itertools
import matplotlib.pyplot as plt

# Go to current directory
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

# Define experiment parameters

# general
keyboard = "qwerty" # azerty
pilot = 1 #Do we want a reduced number of trials for piloting????
training = 1
StairCase = 1 #If set to 0 no Staircase
instructions = 1
fullscreen = 1 #Developper mode
Part = 1 # 0 = Part 1 and 1 = Part 2

# training
if not pilot: 
    n_training = 20 # number of training trials per block
else:
    n_training = 5

################################################################################################################################ 
#Staircase variables
################################################################################################################################   
per_correct = 0 # percentage correct when starting --> for training if percetage is lower start another block 
des_per_cor = 0.7 # desired percentage correct
mean_rt = 5 # mean reaction time before starting (seconds)
des_mean_rt = 1.5 # desired mean reaction time --> for training if percetage is lower start another block 
coherence = .80 # coherence for first training block 
coherence_hard = .40 # coherence for second training block (staircase sets it for main exp)

max_dur_conf = 3

# staircases
num_staircase = 1 # set number of staircases to run (QUEST+)
breaktrials = 45 # offer a break during the staircases every n trials
################################################################################################################################ 

# main blocks
if not pilot:
    n_trials = 45 # Number of trials per testing block Note: this should be an even number
    n_blocks = 8 # Number of testing blocks (15)
else:
    n_trials = 25
    n_blocks = 4
block = 0 # starting block number (for training/staircase)

# stimulus
dotLife = 5 

# waiting times (in seconds)
if not pilot:
    ins_wait = 1; break_wait = 5
elif pilot:
    ins_wait = 0; break_wait = 0

# TrialHandler: make a data file -> 
if pilot:
    sub = 0; age = 30; gender = 'Man'; handedness = 'Right'
    info = {"sub": sub, "age": age, "gender": gender, "handedness": handedness} #creating a dictionnary
else:
    info = {"Subject number": 0, "gender": ['Woman', 'Man', 'X'], "age": 0, "handedness": ['Left', 'Right']}
    myDlg = gui.DlgFromDict(dictionary=info, title="DotsTask", show=True)
    sub = info['Subject number']; age = info['age']; gender = info['gender']; handedness = info['handedness']
    
file_name = "Data/RDM_reportz_sub%d" % sub
thisExp = data.ExperimentHandler(dataFileName=file_name, extraInfo=info)  # saving extra info along with the main experimental data

# Window set up
if fullscreen:
    win = vis.Window(size=[1536,960], units = 'pix', color='black', allowGUI=False, fullscr=True) #set fullscr to True for the experiment
else:
    win = vis.Window(size = [600,400], units = 'pix', color='black', allowGUI=False, fullscr=False)
         
win.mouseVisible = False
mouse = event.Mouse(win=win) #Create a mouse object for the slider
#screen_size = pyautogui.size() #Get screen size to start mouse in same position in slider
width = win.size[0]
height = win.size[1]

# Clock
clock = core.Clock()

# Define keys
if keyboard == "qwerty":
    choice_keys = ['q', 'e', 'escape', 'return']  # left, right, escape, enter
elif keyboard == "azerty":
     choice_keys = ['a', 'e', 'escape', 'return']  # left, right, escape, enter
else:
     raise TypeError('Unknown keyboard name')

# Creating DotMotion stimulus
DotMotion = vis.DotStim(win, units='pix', nDots= 120, fieldSize = 300, fieldShape='circle', dotSize=6  , dotLife=10, speed=0.7, color='white', 
                        signalDots='same', noiseDots='walk', ) #https://www.psychopy.org/api/visual/dotstim.html 

# Creating a slider to rate caution 
slider = vis.Slider(win, name='slider', size=(400,20), pos = (0,0), units = 'pix',
                          ticks=(0,1), granularity = 0.01,
                          style=['rating'], color='LightGray', font='HelveticaBold', flip=False)
slider.marker.color = "blue"
slider.marker.size = 20                       
slider_label_wrong = vis.TextStim(win, text= "definitely wrong", pos=(-200, 30)) 
slider_label_right = vis.TextStim(win, text= "definitely right", pos=(200, 30)) 
mouse = event.Mouse(win=win) #Create a mouse object for the slider

# creatiung a fixation cross
fixation = vis.ShapeStim(
    win=win,
    vertices=((0, -20), (0, 20), (0,0), (-20,0), (20,0)),  # vertical and horizontal lines
    lineWidth=5,
    closeShape=False,
    lineColor='white'
)
#Text for confidence no response
no_response_text = vis.TextStim(win, text="No response, try to be faster next trial!",
            color='white', height=30)


width = win.size[0]
height = win.size[1]

slider_instructions = vis.TextStim(win, text = "How confident were you in your decision?", pos=(0,75))

# reading instructions slides -> change instructions!!!
intro = vis.ImageStim(win, image=dname+"\Intro.jpg", units = 'pix', size = [width*0.8,height*0.8])
staircase = vis.ImageStim(win, image=dname+"\Staircase.jpg", units = 'pix', size = [width*0.8,height*0.8])
main_1 = vis.ImageStim(win, image=dname+"\Main1.jpg", units = 'pix', size = [width*0.8, height*0.8]) 
main_2 = vis.ImageStim(win, image=dname+"\Main2.jpg", units = 'pix', size = [width*0.8, height*0.8]) 
main_3 = vis.ImageStim(win, image=dname+"\Main3.jpg", units = 'pix', size = [width*0.8, height*0.8])
main_4 = vis.ImageStim(win, image=dname+"\Main4.jpg", units = 'pix', size = [width*0.8, height*0.8])
main_5 = vis.ImageStim(win, image=dname+"\Main5.jpg", units = 'pix', size = [width*0.8, height*0.8])   

# Functions 
## Function to check if the mouse is hovering over the slider bar area
def is_mouse_over_slider(mouse, slider):
    # Get the mouse position and slider position/size
    mouse_pos = mouse.getPos()
    slider_x, slider_y = slider.pos
    slider_width, slider_height = slider.size
    
    # Check if the mouse is within the horizontal bounds of the slider
    within_x = slider_x - slider_width / 2 <= mouse_pos[0] <= slider_x + slider_width / 2
    # Check if the mouse is within the vertical bounds of the slider (adjust this based on the height)
    within_y = slider_y - slider_height / 2 <= mouse_pos[1] <= slider_y + slider_height / 2
    
    return within_x and within_y

## Function to handle slider value calculation based on active slider
def get_slider_value(mouse_pos, min_value, max_value):
    # Convert normalized mouse position to slider values
    # Assuming `mouse_pos[0]` is between -0.4 and 0.4 for the slider's widyrth
    normalized_position = (mouse_pos[0] + slider.size[0] / 2) / slider.size[0]  # Normalize between 0 and 1
    slider_value = min_value + normalized_position * (max_value - min_value)
    
    # Ensure slider_value stays within the bounds of the slider
    slider_value = max(min(slider_value, max_value), min_value)
    
    return slider_value

# Start experiment ------------------------------------------------------------

win.mouseVisible = False

# welcome text
intro.draw(); win.flip();core.wait(ins_wait); event.waitKeys(keyList=['space'])

# training blocks
if training:
    TrialType = "Training - easy"
    while (per_correct <= des_per_cor or mean_rt > des_mean_rt):
        print("Conditions not met, starting another practice block")
        # randomize left and right movement (0 = left, 1 = right)

        # training: 50% left and right
        condition_direction = np.repeat(range(2),[math.floor(n_training*0.5), math.ceil(n_training*0.5)]); random.shuffle(condition_direction) #Creates an equal amount of left/right trials

        #Empty lists of accuracy and reaction times for training data  
        acc = [0] * n_training 
        rt = [0] * n_training
        for trial in range(n_training):
            # Stimulus direction
            if condition_direction[trial] == 0:
                print(condition_direction[trial])
                correct = 'left'; direction = 180
            if condition_direction[trial] == 1:
                correct = 'right'; direction = 0 
            
            # draw stimulus
            resp = [] #empty list for response
            event.clearEvents() 
            DotMotion.coherence = coherence
            DotMotion.dotLife = dotLife
            DotMotion.dir = direction

            # save start time of the stimulus    
            T_stimulus_start = clock.getTime()
            while len(resp) == 0:
                DotMotion.draw()
                win.flip()
                resp = event.getKeys(keyList=choice_keys)
                
                if clock.getTime() - T_stimulus_start > des_mean_rt:
                    print("No response within 1.5 s, skipping trial")
                    resp.append("No resp") # mark as no response
                    FB_text = "No response"
                    FB_col = "grey"
                    
                 
            if resp:
                T_stimulus_stop = clock.getTime()
                RTdec = T_stimulus_stop - T_stimulus_start
                rt[trial] = RTdec
                print("Reaction time is:", RTdec)
            else:
                RTdec = np.nan
                rt[trial] = RTdec
        
            
            # get accuracy
            if resp:
                if correct == 'left' and resp[0] == choice_keys[0]:
                    ACC = 1; acc[trial] = 1
                    print("Decision was correct")
                    FB_text = "Correct!"; FB_col = 'green'
                    
                elif correct == 'right' and resp[0] == choice_keys[0]:
                    ACC = 0; acc[trial] = 0
                    print("Decision was incorrect")
                    FB_text = "Wrong"; FB_col = 'red'
                elif correct == 'left' and resp[0] == choice_keys[1]:
                    ACC = 0; acc[trial] = 0
                    print("Decision was incorrect")
                    FB_text = "Wrong"; FB_col = 'red'
                elif correct == 'right' and resp[0] == choice_keys[1]:
                    ACC = 1; acc[trial] = 1
                    print("Decision was correct")
                    FB_text = "Correct!"; FB_col = 'green'
                else:
                    ACC = 0
            
            # allow escape to exit experiment
            if resp == ['escape']:
                print('Participant pressed escape')
                thisExp.saveAsWideText(file_name + '.csv', delim=',') 
                win.close()
                core.quit()
            
            # Give feedback
            feedback = vis.TextStim(win, text = FB_text, color = FB_col, height=40)
            feedback.draw()
            win.flip()
            if resp:
                core.wait(0.5)
            else:
                core.wait(1)
            
            if resp[0] == choice_keys[1]:
                resp[0] = "right"
            elif resp[0] == choice_keys[0]:
                resp[0] = "left"
            
            thisExp.addData("block", block)
            thisExp.addData("Trialtype", TrialType)
            thisExp.addData("withinblocktrial", trial)
            thisExp.addData("RTdec", RTdec)
            thisExp.addData("resp", resp)
            thisExp.addData("cor", ACC)
            thisExp.addData("dots direction", direction)
            thisExp.addData("cor_resp", correct)
            thisExp.addData("coherence", DotMotion.coherence)
            thisExp.nextEntry()
            
        per_correct = sum(acc) / len(acc)
        mean_rt = np.nanmean(rt)
        print("Percent correct is:", per_correct)
        print("Average reaction time is:", mean_rt)
        if (per_correct >= des_per_cor and mean_rt <= des_mean_rt and TrialType == "Training - easy"):
            print("Conditions met, moving on to harder training block")
            
            feedback_text = vis.TextStim(win, text = "Great job! We're going to make it a little more difficult now.")
            break_text = vis.TextStim(win, text = "Take a short break before we continue with the next block.", pos = (0,-100))
            space = vis.TextStim(win, text='Press space to continue', pos=(0, -150), height=20)
            feedback_text.draw(); break_text.draw(); win.flip()
            core.wait(break_wait); feedback_text.draw(); break_text.draw(); space.draw(); win.flip(); event.waitKeys(keyList=['space']) #Use corewait for intertrial interval
                
            TrialType = "Training - hard"
            coherence = coherence_hard
            per_correct = 0; mean_rt = 2; # set back to default for new run
            
            
        elif (per_correct >= des_per_cor and mean_rt <= des_mean_rt and TrialType == "Training - hard"):
            print("Conditions met, moving on to real experiment")
            break
        
        elif (per_correct <= des_per_cor or mean_rt > des_mean_rt):
               
            # offer a break
            points_text = vis.TextStim(win, text = 'You got ' + str(per_correct*100) + '% correct.', pos=(0, 100))
            speed_text = vis.TextStim(win, text = 'Your average reaction time was ' + str(f"{mean_rt:.2f}") + " seconds", pos = (0,50))
        
            if per_correct < des_per_cor and mean_rt > des_mean_rt:
                feedback_text = vis.TextStim(win, text = 'Try to be faster and more accurate in the next block!')
            elif per_correct < des_per_cor and mean_rt <= des_mean_rt:
                feedback_text = vis.TextStim(win, text = 'Try to be more accurate in the next block!')
            elif per_correct >= des_per_cor and mean_rt > des_mean_rt:
                feedback_text = vis.TextStim(win, text = 'Try to be faster in the next block!')
        
            break_text = vis.TextStim(win, text = "Take a short break before we continue with the next block.", pos = (0,-100))
            space = vis.TextStim(win, text='Press space to continue', pos=(0, -150), height=20)
            points_text.draw(); speed_text.draw(); feedback_text.draw(); break_text.draw(); win.flip()
            core.wait(break_wait); points_text.draw(); speed_text.draw(); feedback_text.draw(); break_text.draw(); space.draw(); win.flip(); event.waitKeys(keyList=['space'])          
    
if instructions:
    # instructions on real experiment
    # break_text = vis.TextStim(win, text = "We will now continue with the main experiment")
    # space = vis.TextStim(win, text='Press space to continue', pos=(0, -50), height=20)
    # break_text.draw(); win.flip()
    # core.wait(5); break_text.draw(); space.draw(); win.flip(); event.waitKeys(keyList=['space'])
    main_1.draw(); win.flip();core.wait(ins_wait); event.waitKeys(keyList=['space'])  
    main_2.draw(); win.flip();core.wait(ins_wait); event.waitKeys(keyList=['space']) 
    main_3.draw(); win.flip();core.wait(ins_wait); event.waitKeys(keyList=['space']) 
    main_4.draw(); win.flip();core.wait(ins_wait); event.waitKeys(keyList=['space'])  
    main_5.draw(); win.flip();core.wait(ins_wait); event.waitKeys(keyList=['space']) 

#Variables for main experiment
inter_t_mean = 2.5 # Mean inter trial interval (ITI)
inter_t_sd = 1 # SD of ITI
lower = 1 # Lower cutoff for trunc nom
upper = 4 # Upper cutoff for trunc nom
means = [2.375, 4.125] #calculated so symetrical 
sds = [0.05, 0.5]
pairs = list(itertools.product(means, sds))
repeat = n_blocks // len(pairs)
conditions = repeat * pairs
np.random.shuffle(conditions)
print(conditions)

def standard(low, mean,sd):
    Z = (low-mean)/sd
    return Z


a = standard(lower,inter_t_mean,inter_t_sd)#standardized cut offs
b = standard(upper, inter_t_mean,inter_t_sd)
# Create staircase
SC = qp.QuestPlus(stim_domain= {"intensity": np.linspace(0.01,1,50)},
                 func="weibull",
                 stim_scale="log10",
                 param_domain= {"threshold": np.linspace(0.01,1,50), "slope": np.linspace(1,10,50), "lower_asymptote": 0.5, "lapse_rate": np.array([0.01,0.03,0.05,0.07,0.1])},
                 prior = {"threshold": np.ones(50)/50, "slope": np.ones(50)/50, "lapse_rate": np.repeat(0.2, 5)},
                 outcome_domain={"response": [1, 0]},
                 stim_selection_method="min_n_entropy",
                 stim_selection_options = {"n": 1, "max_consecutive_reps": 2},
                 param_estimation_method= "mean")

#data.QuestPlusHandler(nTrials = n_trials * n_blocks, psychometricFunc = "weibull", intensityVals = np.linspace(0.01,1,50),
                                    #thresholdVals = np.linspace(0.01,1,50), slopeVals = np.linspace(1, 10, 50), lowerAsymptoteVals = [0.5], 
                                    #lapseRateVals = [0.01], responseVals = [1,0], prior = {"threshold":  np.ones(50)/50, "slope": np.ones(50)/50},
                                    #stimScale = "linear", stimSelectionMethod = "minNEntropy", 
                                    #stimSelectionOptions = {"N": 1, "maxConsecutiveReps": 2}, 
                                    #paramEstimationMethod = "mean")
# testing blocks
TrialType = "Main"
# Equal # trials left and right for first block 
condition_direction = np.repeat(range(2),[math.floor(n_trials*0.5), math.ceil(n_trials*0.5)]); random.shuffle(condition_direction)
 # determine waiting times between trials and waiting times confidence interval for first block
inter_trial = truncnorm.rvs(a, b, loc=inter_t_mean, scale=inter_t_sd, size= n_trials) #Can change mean according to pilots

if not Part:
    manipulation = np.random.uniform(low = des_mean_rt, high = 5, size = n_trials) #Based on Bradley et al. (2012) "Orienting and Emotional Perception: Facilitation, Attenuation, and Interference"
else:
    current_mean = conditions[0][0]
    print(current_mean)
    current_sd = conditions[0][1]
    print(current_sd)
    a2 = standard(1.5,current_mean, current_sd)
    b2 = standard(5, current_mean, current_sd)
    manipulation = truncnorm.rvs(a2, b2, loc= current_mean, scale=current_sd, size= n_trials) 

#Initiate vectors for first block 
acc = [0] * n_trials
rt = [0] * n_trials

trialN = 0
blockN = 0
#Initiate staircase 
for eachTrial in range(n_trials*n_blocks):
    print(trialN)
    print(eachTrial)

    # Stimulus direction
    if condition_direction[trialN] == 0:
        correct = 'left'; direction = 180
    if condition_direction[trialN] == 1:
        correct = 'right'; direction = 0
            
    # save start time of the stimulus    
    T_stimulus_start = clock.getTime()
    
    # draw stimulus
    coherence = SC.next_stim
    print(coherence)
    resp = []
    event.clearEvents() 
    DotMotion.coherence = coherence["intensity"]
    DotMotion.dotLife = dotLife
    DotMotion.dir = direction
    while len(resp) == 0:
        DotMotion.draw()
        win.flip()
        resp = event.getKeys(keyList=choice_keys)

        if clock.getTime() - T_stimulus_start > des_mean_rt:
            print("No response within 1.5 s, skipping trial")
            resp.append("No resp")  # mark as no response
            miss_text = vis.TextStim(win, text = "No response, try to be faster next trial!", height = 30)
            miss_text.draw(); win.flip()
            core.wait(1)
            break
            
    #clear screen
    fixation.draw() 
    win.flip()

    # get reaction time
    if resp:
        T_stimulus_stop = clock.getTime()
        RTdec = T_stimulus_stop - T_stimulus_start
        rt[trialN] = RTdec
        print("Reaction time is:", RTdec)
    else:
        rt[trialN] = np.nan
        
        # get accuracy
    if resp:
        if correct == 'left' and resp[0] == choice_keys[0]:
            ACC = 1; acc[trialN] = 1
            print("Decision was correct")
        elif correct == 'right' and resp[0] == choice_keys[0]:
            ACC = 0; acc[trialN] = 0
            print("Decision was incorrect")
        elif correct == 'left' and resp[0] == choice_keys[1]:
            ACC = 0; acc[trialN] = 0
            print("Decision was incorrect")
        elif correct == 'right' and resp[0] == choice_keys[1]:
            ACC = 1; acc[trialN] = 1
            print("Decision was correct")
        else: 
            ACC = 0
            
            # allow escape to exit experiment
    if resp == ['escape']:
        print('Participant pressed escape')
        thisExp.saveAsWideText(file_name + '.csv', delim=',') 
        win.close()
        core.quit()
    

        #Waiting time for cofidence ratings
    if resp:
        interval = manipulation[trialN]
        core.wait(interval - rt[trialN])
        print("interval = ", manipulation[trialN])
        T_rating_start = clock.getTime()
        win.mouseVisible = True
        mouse.setPos([0,0])
        slider.reset()
        slider.draw()
        slider_instructions.draw()
        slider_label_wrong.draw()
        slider_label_right.draw()
        win.flip()

        SR_conf = None
        while SR_conf is None: 
            # check if participant is 
            elapsed_time = clock.getTime() - T_rating_start
            if elapsed_time > max_dur_conf:
                win.flip(clearBuffer=True)
                no_response_text.draw()
                win.flip()
                core.wait(1)
                SR_conf = None  # make sure response is recorded as missing
                RTrating = None
                break
            # Check if the mouse is over the slider bar area 
            if is_mouse_over_slider(mouse, slider):
                mouse_pos = mouse.getPos()
                #print("Mouse position is: ", mouse_pos)
                slider_pract_value = get_slider_value(mouse_pos, 0, 1)
                #print("slider marker pos = ", slider_pract_value)
                slider.markerPos = slider_pract_value  # Update slider marker position
        
                # Redraw the slider and instructions
                slider.draw()
                slider_instructions.draw()
                slider_label_wrong.draw()
                slider_label_right.draw()
                win.flip()

            # Check if the mouse has been clicked to submit the answer
            if mouse.getPressed()[0] & is_mouse_over_slider(mouse, slider):
                SR_conf = (slider.markerPos - 0.5) * 2 # Get the final rating value
                win.mouseVisible = False
                print("Reported confidence = ", SR_conf)
                T_rating_stop = clock.getTime()
                RTrating = T_rating_stop - T_rating_start
                
            # Escape 
            keys = event.getKeys()  
            if 'escape' in keys:  
                print('Participant pressed escape')
                thisExp.saveAsWideText(file_name + '.csv', delim=',') 
                win.close()  
                core.quit()
        
    else:
        RTrating = None
        SR_conf = None
        interval = None

    #Add response to staircase and proceed to next value
    SC.update(stim= coherence, outcome= {"response":ACC})
        

    # Blank screen drawn from a truncated normal distribution
    fixation.draw(); win.flip(); core.wait(inter_trial[trialN]) # Change to waiting time drawn from a distribution 
    
    if resp[0] == choice_keys[1]:
        resp[0] = "right"
    else: 
        resp[0] = "left"
    # Save trial
    thisExp.addData("block", blockN + 1)
    thisExp.addData("Trialtype", TrialType)
    thisExp.addData("withinblocktrial" , trialN + 1)
    thisExp.addData("RTdec", RTdec)
    thisExp.addData("resp", resp)
    thisExp.addData("cor", ACC)
    thisExp.addData("dots direction", direction)
    thisExp.addData("cor_resp", correct)
    thisExp.addData("interval", interval)
    thisExp.addData("interTrial.interval", inter_trial[trialN])
    if Part:
        thisExp.addData("Mean", conditions[blockN][0])
        thisExp.addData("Standard deviation", conditions[blockN][1])
    thisExp.addData("SR_conf", SR_conf)
    thisExp.addData("RTrating", RTrating)
    thisExp.addData("coherence", coherence["intensity"])
    thisExp.nextEntry()
    
    #Update trialN
    trialN += 1
    
    if blockN < n_blocks - 1:
    #Check if next block     
        if trialN == n_trials: 
            blockN += 1
            trialN = 0
            #Update variables for next block
            condition_direction = np.repeat(range(2),[math.floor(n_trials*0.5), math.ceil(n_trials*0.5)]); random.shuffle(condition_direction)
            # determine waiting times between trials and waiting times confidence interval
            inter_trial = truncnorm.rvs(a, b, loc=inter_t_mean, scale=inter_t_sd, size= n_trials) #Can change mean according to pilots

            if not Part:
                manipulation = np.random.uniform(low = des_mean_rt, high = 5, size = n_trials) #Based on Bradley et al. (2012) "Orienting and Emotional Perception: Facilitation, Attenuation, and Interference"
            else:
                current_mean = conditions[blockN][0]
                print(current_mean)
                current_sd = conditions[blockN][1]
                print(current_sd)
                a2 = standard(1.5,current_mean, current_sd)
                b2 = standard(5, current_mean, current_sd)
                manipulation = truncnorm.rvs(a2, b2, loc= current_mean, scale=current_sd, size= n_trials) 

            SC.stim_selection_options["n"] += 10
            print(SC.stim_selection_options["n"])
            #performance for block 
            num_correct = sum(acc) 
            tot_trials = len(acc)
            per_correct = sum(acc)/len(acc)
            mean_rt = np.nanmean(rt)
            print("Percent correct is:", per_correct)
            print("Average reaction time is:", mean_rt)
                
            #Initiate vectors 
            acc = [0] * n_trials
            rt = [0] * n_trials

            # offer a break
            points_text = vis.TextStim(win, text = 'You got ' + str(num_correct) + ' out of ' + str(tot_trials) + ' points this block!', pos=(0, 100))
            speed_text = vis.TextStim(win, text = 'Your average reaction time was ' + str(f"{mean_rt:.2f}") + " seconds", pos = (0,50))
            
            if per_correct < des_per_cor and mean_rt > des_mean_rt:
                feedback_text = vis.TextStim(win, text = 'Try to be faster and more accurate in the next block!')
            elif per_correct < des_per_cor and mean_rt <= des_mean_rt:
                feedback_text = vis.TextStim(win, text = 'Try to be more accurate in the next block!')
            elif per_correct >= des_per_cor and mean_rt > des_mean_rt:
                feedback_text = vis.TextStim(win, text = 'Try to be faster in the next block!')
            elif per_correct >= des_per_cor and mean_rt <= des_mean_rt:
                feedback_text = vis.TextStim(win, text = 'Good job! Try to be even faster and more accurate!')

            
            break_text = vis.TextStim(win, text = "Take a short break before we continue with the next block (block " + str(blockN+1) + "/" + str(n_blocks) + ")", pos = (0, -100))
            space = vis.TextStim(win, text='Press space to continue', pos=(0, -150), height=20)
            points_text.draw(); speed_text.draw(); feedback_text.draw(); break_text.draw(); win.flip()
            core.wait(break_wait); points_text.draw(); speed_text.draw(); feedback_text.draw(); break_text.draw(); space.draw(); win.flip(); event.waitKeys(keyList=['space'])
###########################################################################################################################################################################################      
# End of the experiment
break_text = vis.TextStim(win, text = "This is the end of the experiment. \n Thank you very much for your participation!")
space = vis.TextStim(win, text='Press space to close the experiment', pos=(0, -50), height=20)
break_text.draw(); win.flip()
core.wait(break_wait); break_text.draw(); space.draw(); win.flip(); event.waitKeys(keyList=['space'])
        
# Save data in a csv file -----------------------------------------------------
thisExp.saveAsWideText(file_name + '.csv', delim=',') 
  
# End of the experiment -------------------------------------------------------
win.close()
core.quit() 