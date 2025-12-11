from psychopy import visual as vis
from psychopy import core, event
import numpy as np

######################################################################
#Introduction
######################################################################
win = vis.Window(size=[1536,960], units = 'pix', color='grey', allowGUI=False, fullscr=True) 

welcome_text = vis.TextStim(win, text = "Welcome to this experiment!", pos=(0,300), height = 55, color = 'white', wrapWidth=2000 )
main_text = vis.TextStim(
    win,
    text=(
        "Today, you will be looking at moving dots and deciding whether you think\n"
        "they are moving mostly up or mostly down. We will practice this now.\n" 
        "Place your left middle finger on the “q” key and your left index\n"
        "finger on the “a” key. If you think the dots are moving up, press the\n"
        "“q”, and if you think they are moving down, press the “a”\n"
        "You will get feedback about your decision.\n"
        "Try to be both fast and accurate."
    ),
    pos=(0, 50) ,
    height= 40,
    color='white',
    wrapWidth=1400,
    alignText='left'
)
img = vis.ImageStim(
    win,
    image= r"C:\Users\User\OneDrive\Documenten\psychologie 1e Master\internship\RDM_exp\keyboard.png",   
    pos=(350, -150),              
    size=(500, 250)             
)

hand = vis.ImageStim(
    win,
    image= r"C:\Users\User\OneDrive\Documenten\psychologie 1e Master\internship\RDM_exp\hand.png",   
    pos=(300, -300),              
    size=(175,175)             
)

space_text = vis.TextStim(
    win,
    text= "Press SPACE to continue.",
    pos=(0, -350) ,
    height= 35,
    color='white',
    wrapWidth=1400,
    alignText='left'
)

welcome_text.draw()
main_text.draw()
img.draw()
hand.draw()
space_text.draw()
win.flip()

win.getMovieFrame(buffer='front')  
win.saveMovieFrames(r'C:\Users\User\OneDrive\Documenten\psychologie 1e Master\internship\RDM_exp\Intro.jpg')

print("Saved intro screen to:")
print(r'C:\Users\User\OneDrive\Documenten\psychologie 1e Master\internship\RDM_exp\Intro.jpg')

while True:
    keys = event.getKeys()
    if 'space' in keys:
        break


######################################################################
#Main 1
######################################################################
text_1 = vis.TextStim(win, text = "Good job! You seem to get the hang of the dots.", pos=(0,150), height = 40, color = 'white', wrapWidth=2000 )
text_2 = vis.TextStim(win, text = "We will now introduce something new to the task.", pos=(0,0), height = 40, color = 'white', wrapWidth=2000 )

space_text = vis.TextStim(
    win,
    text= "Press SPACE to continue.",
    pos=(0, -250) ,
    height= 30,
    color='white',
    wrapWidth=1400,
    alignText='center'
)

text_1.draw()
text_2.draw()
space_text.draw()
win.flip()

win.getMovieFrame(buffer='front')  
win.saveMovieFrames(r'C:\Users\User\OneDrive\Documenten\psychologie 1e Master\internship\RDM_exp\Main1.jpg')

print("Saved intro screen to:")
print(r'C:\Users\User\OneDrive\Documenten\psychologie 1e Master\internship\RDM_exp\Main1.jpg')

while True:
    keys = event.getKeys()
    if 'space' in keys:
        break

######################################################################
#Main 2
######################################################################
text_3 = vis.TextStim( win=win,
    text="In the next blocks, we will ask you to report how confident you are about your decisions. After your choice, you will see a scale like this:",
    pos=(0, 300),color='white',height= 40,wrapWidth=1200, alignText='center'
)

confidence_question = vis.TextStim(win, text = "How confident were you in your decision?", pos=(0,100), height = 30, wrapWidth=1200)

space_text = vis.TextStim(
    win,
    text= "Press SPACE to continue.",
    pos=(0, -325),
    height = 30,
    color='white',
    wrapWidth=1400,
    alignText='center'
)

slider = vis.Slider(win, name='slider', size=(600,40), pos = (0,-20), units = 'pix',
                          ticks=(0,1), granularity = 0.01,
                          style=['rating'], color='black',lineColor='white',    # line/bar color  ← you need this
    fillColor='white', font='HelveticaBold', flip=False)


slider.marker.color = "white"
slider.markerPos = 0.5 
slider.marker.size = 30                       
slider_label_wrong = vis.TextStim(win, text= "definitely wrong", pos=(-300, 30), color = "white", height=23) 
slider_label_right = vis.TextStim(win, text= "definitely right", pos=(300, 30), color = "white", height=23)

text_4 = vis.TextStim( win=win,
    text="We would like you to report how confident you are in your choice.",
    pos=(0, -175),color='white',height= 40,wrapWidth=1200, alignText='center'
)

text_3.draw()
text_4.draw()
space_text.draw()
confidence_question.draw()
slider.draw()
slider_label_right.draw()
slider_label_wrong.draw()
win.flip()

win.getMovieFrame(buffer='front')  
win.saveMovieFrames(r'C:\Users\User\OneDrive\Documenten\psychologie 1e Master\internship\RDM_exp\Main2.jpg')

print("Saved intro screen to:")
print(r'C:\Users\User\OneDrive\Documenten\psychologie 1e Master\internship\RDM_exp\Main2.jpg')

while True:
    keys = event.getKeys()
    if 'space' in keys:
        break

######################################################################
#Main 3
######################################################################
text_5 = vis.TextStim( win=win,
    text="You can move the white cursor with the mouse in your right hand, and click to confirm your choice. " \
    "For example, if you felt really confident that your last decision was correct, you might click the scale like this:",
    pos=(0, 300),color='white',height= 40,wrapWidth=1400, alignText='left'
)

slider = vis.Slider(win, name='slider', size=(400,20), pos = (0,75), units = 'pix',
                          ticks=(0,1), granularity = 0.01,
                          style=['rating'], color='white', font='HelveticaBold', flip=False)
slider.marker.color = "white"
slider.marker.size = 20     
slider.markerPos = 0.9                  
slider_label_wrong = vis.TextStim(win, text= "definitely wrong", pos=(-200, 105)) 
slider_label_right = vis.TextStim(win, text= "definitely right", pos=(200, 105)) 
slider_instructions = vis.TextStim(win, text = "How confident were you in your decision?", pos=(0,150))

text_6 = vis.TextStim( win=win,
    text="But if you are really certain that you made a wrong decision, you might do this:",
    pos=(0, -75),color='white',height= 40,wrapWidth=1400, alignText='left'
)

slider_2 = vis.Slider(win, name='slider', size=(400,20), pos = (0,-250), units = 'pix',
                          ticks=(0,1), granularity = 0.01,
                          style=['rating'], color='white', font='HelveticaBold', flip=False)
slider_2.marker.color = "white"
slider_2.marker.size = 20     
slider_2.markerPos = 0.1                  
slider_label_wrong_2 = vis.TextStim(win, text= "definitely wrong", pos=(-200, -220)) 
slider_label_right_2 = vis.TextStim(win, text= "definitely right", pos=(200, -220)) 
slider_instructions_2 = vis.TextStim(win, text = "How confident were you in your decision?", pos=(0,-175))

space_text = vis.TextStim(
    win,
    text= "Press SPACE to continue.",
    pos=(0, -350) ,
    height= 30,
    color='white',
    wrapWidth=1400,
    alignText='center'
)

text_5.draw()
slider.draw()
slider_label_right.draw()
slider_label_wrong.draw()
slider_instructions.draw()
text_6.draw()
slider_2.draw()
slider_label_right_2.draw()
slider_label_wrong_2.draw()
slider_instructions_2.draw()
space_text.draw()
win.flip()

win.getMovieFrame(buffer='front')  
win.saveMovieFrames(r'C:\Users\User\OneDrive\Documenten\psychologie 1e Master\internship\RDM_exp\Main3.jpg')

print("Saved intro screen to:")
print(r'C:\Users\User\OneDrive\Documenten\psychologie 1e Master\internship\RDM_exp\Main3.jpg')

while True:
    keys = event.getKeys()
    if 'space' in keys:
        break 
######################################################################
#Main 4
######################################################################
text_7 = vis.TextStim( win=win,
    text="Try to use the entire scale, even if you feel like you were guessing on your last decision." \
    " If you dont feel confident at all, you should answer around the middle of the scale, like this:",
    pos=(0, 300),color='white',height= 40,wrapWidth=1400, alignText='left'
)

slider = vis.Slider(win, name='slider', size=(600,40), pos = (0,-20), units = 'pix',
                          ticks=(0,1), granularity = 0.01,
                          style=['rating'], color='white', font='HelveticaBold', flip=False)
slider.marker.color = "white"
slider.marker.size = 30     
slider.markerPos = 0.5                  
slider_label_wrong = vis.TextStim(win, text= "definitely wrong", pos=(-300, 30), height = 23) 
slider_label_right = vis.TextStim(win, text= "definitely right", pos=(300, 30), height = 23) 
slider_instructions = vis.TextStim(win, text = "How confident were you in your decision?", pos=(0,100), height = 30, wrapWidth=1200)
guessing =  vis.TextStim(win, text= "guessing", pos=(0, 30), height = 23) 

triangle = vis.ShapeStim(
    win,
    vertices=[(-10, -5), (10, -5), (0, 10)],  
    fillColor='red', lineColor='red'
)
triangle.pos = (0, -60)

space_text = vis.TextStim(
    win,
    text= "Press SPACE to continue.",
    pos=(0, -325),
    height = 30,
    color='white',
    wrapWidth=1400,
    alignText='center'
)

text_7.draw()
space_text.draw()
slider.draw()
slider_label_wrong.draw()
slider_label_right.draw()
slider_instructions.draw()
triangle.draw()
guessing.draw()
win.flip()

win.getMovieFrame(buffer='front')  
win.saveMovieFrames(r'C:\Users\User\OneDrive\Documenten\psychologie 1e Master\internship\RDM_exp\Main4.jpg')

print("Saved intro screen to:")
print(r'C:\Users\User\OneDrive\Documenten\psychologie 1e Master\internship\RDM_exp\Main4.jpg')

while True:
    keys = event.getKeys()
    if 'space' in keys:
        break

######################################################################
#Extra Scale 
######################################################################
text_8 = vis.TextStim(win=win,
    text="In some trials we will ask you a different question after your decision. " \
    "On this scale, you have to indicate how clear the direction of the moving dots was, like this: ",
    pos=(0, 300),color='white',height= 38,wrapWidth=1400, alignText='left'
)

slider = vis.Slider(win, name='slider', size=(600,40), pos = (0,0), units = 'pix',
                          ticks=(0,1), granularity = 0.01,
                          style=['rating'], color='white', font='HelveticaBold', flip=False)
slider.marker.color = "white"
slider.marker.size = 30     
slider.markerPos = 0.5                  
slider_label_wrong = vis.TextStim(win, text= "Not clear at all", pos=(-300, 50), height = 23) 
slider_label_right = vis.TextStim(win, text= "Very clear", pos=(300, 50), height = 23) 
instr = vis.TextStim(win, text = "How clear was the DIRECTION of the dots?", pos=(0,120), height = 30, wrapWidth=1200)
line = vis.Line(win, start=(-40, 100), end=(125, 100), lineWidth=3, lineColor="white")

space_text = vis.TextStim(
    win,
    text= "Press SPACE to continue.",
    pos=(0, -350),
    height = 30,
    color='white',
    wrapWidth=1400,
    alignText='center'
)

text_9 = vis.TextStim( win=win,
    text="The scale works in the same way but now you have to evaluate the clarity of the motion direction.\n" 
    "PAY ATTENTION to which question you are giving an answer to when using the scale!",
    pos=(0, -180),color='white',height= 38,wrapWidth=1400, alignText='left'
)


text_8.draw()
space_text.draw()
slider.draw()
slider_label_wrong.draw()
slider_label_right.draw()
instr.draw()
line.draw()
text_9.draw()
win.flip()

win.getMovieFrame(buffer='front')  
win.saveMovieFrames(r'C:\Users\User\OneDrive\Documenten\psychologie 1e Master\internship\RDM_exp\ExtraScale.jpg')

print("Saved intro screen to:")
print(r'C:\Users\User\OneDrive\Documenten\psychologie 1e Master\internship\RDM_exp\ExtraScale.jpg')

while True:
    keys = event.getKeys()
    if 'space' in keys:
        break

######################################################################
#Main 5
######################################################################
text_10 = vis.TextStim(win=win,
    text="If everything is clear, we will give you some training with the scales. Remember, use the 'q' and 'a' key to report your decision" \
    " and the mouse to click the scale indicating confidence or clarity.",
    pos=(0, 300),color='white',height= 38,wrapWidth=1400, alignText='left'
)

img = vis.ImageStim(
    win,
    image= r"C:\Users\User\OneDrive\Documenten\psychologie 1e Master\internship\RDM_exp\keyboard.png",   
    pos=(0, 80),              
    size=(500, 250)             
)

hand = vis.ImageStim(
    win,
    image= r"C:\Users\User\OneDrive\Documenten\psychologie 1e Master\internship\RDM_exp\hand.png",   
    pos=(-50, -70),              
    size=(175,175)             
)

hand_1 = vis.ImageStim(
    win,
    image= r"C:\Users\User\OneDrive\Documenten\psychologie 1e Master\internship\RDM_exp\hand(1).png",   
    pos=(380, -70),              
    size=(175,175)             
)

mouse = vis.ImageStim(
    win,
    image= r"C:\Users\User\OneDrive\Documenten\psychologie 1e Master\internship\RDM_exp\mouse.png",   
    pos=(300, -70),              
    size=(175,175)             
)

text_11 = vis.TextStim(win=win,
    text="If you have any questions or if something is still unclear, please ask the researcher now. When you are ready, press the SPACE bar to begin the training.",
    pos=(0, -230),color='white',height= 38,wrapWidth=1400, alignText='left'
)

space_text = vis.TextStim(
    win,
    text= "Press SPACE to continue.",
    pos=(0, -350),
    height = 30,
    color='white',
    wrapWidth=1400,
    alignText='center'
)


text_10.draw()
img.draw()
hand.draw()
hand_1.draw()
mouse.draw()
text_11.draw()
space_text.draw()
win.flip()

win.getMovieFrame(buffer='front')  
win.saveMovieFrames(r'C:\Users\User\OneDrive\Documenten\psychologie 1e Master\internship\RDM_exp\Main5.jpg')

print("Saved intro screen to:")
print(r'C:\Users\User\OneDrive\Documenten\psychologie 1e Master\internship\RDM_exp\Main5.jpg')

while True:
    keys = event.getKeys()
    if 'space' in keys:
        break

######################################################################
#Main 5
######################################################################
text_1 = vis.TextStim(win, text = "Good job! You seem to get the hang of the scales.", pos=(0,150), height = 40, color = 'white', wrapWidth=1400 )
text_2 = vis.TextStim(win, text = "You can take a break before starting the main experiment. Press The SPACE bar to continue whenever you are ready to begin. Good Luck!", pos=(0,0), height = 40, color = 'white', wrapWidth=1400)

space_text = vis.TextStim(
    win,
    text= "Press SPACE to continue.",
    pos=(0, -250) ,
    height= 30,
    color='white',
    wrapWidth=1400,
    alignText='center'
)

text_1.draw()
text_2.draw()
space_text.draw()
win.flip()

win.getMovieFrame(buffer='front')  
win.saveMovieFrames(r'C:\Users\User\OneDrive\Documenten\psychologie 1e Master\internship\RDM_exp\Main6.jpg')

print("Saved intro screen to:")
print(r'C:\Users\User\OneDrive\Documenten\psychologie 1e Master\internship\RDM_exp\Main6.jpg')
while True:
    keys = event.getKeys()
    if 'space' in keys:
        break
win.close()




