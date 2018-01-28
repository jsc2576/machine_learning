# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 17:13:18 2017

@author: jsc5565
"""

import os
import tkinter as tk
from PIL import ImageTk, Image
from win32api import GetSystemMetrics
import time
import threading
import cv2
import subprocess
import gc

player_path = ''

check_thread_stop = False

check_extension = ["jpg", "png", "gif", "jpeg"]
check_video_extension = ["avi", "mp4", "mov", "mpeg", "mpg", "mkv"]

def search(dirname):
    file_list = []
    filenames = os.listdir(dirname)
    extension = check_extension + check_video_extension
    for filename in filenames:  
        full_filename = os.path.join(dirname, filename)
        if os.path.isfile(full_filename):
            if any((str(full_filename)).lower()[full_filename.rindex('.')+1:] == ext for ext in extension):
                file_list.append(full_filename)
    return file_list
#search(path_dir)

def ShowImage(file_path):
    image = Image.open(file_path)
    image.show()
    
def PlayVideo(file_path, player_path):
    #video_path = os.path.join(path, file_path)
    subprocess.Popen([player_path, file_path], stdout=subprocess.PIPE)
    #subprocess.call()
    #os.system(player_path+" "+video_path)
    
def changeX(window, width, path, right, speed): # 쓰레드 함수 
    
    file_list = search(path) # 파일 리스트 검색
    image = [] # 이미지 리스트 
    Button = [] # 버튼 리스트
    y_pos = [] # 각 버튼 별 좌표 
    y_height = [] # 각 버튼 별 높
    loop = 0
    y = 0 # 현재 생성된 버튼의 전체 높이 
    
    while True:
        
        if loop >= len(file_list):  
                loop = 0
                
        # 이미지일 경우 처리
        if any((str(file_list[loop])).lower()[file_list[loop].rindex('.')+1:] == ext for ext in check_extension):
            img = Image.open(file_list[loop])
            img_height = int((width*img.size[1])/img.size[0])
            y_height.append(img_height)
            img = img.resize((width, img_height), Image.ANTIALIAS)
            image.append(ImageTk.PhotoImage(img))
            Button.append(tk.Button(window, image=image[loop], bg="white", command=lambda sel=loop: ShowImage(file_list[sel])))
            
        #비디오일 경우 처리 
        else:
            video_img_cv = cv2.VideoCapture(file_list[loop])
            _, video_img = video_img_cv.read()
            #print(video_img)
            video_img = cv2.flip(video_img,1)
            frame_height = video_img.shape[0]
            frame_width = video_img.shape[1]
            img_height = int((width*img.height)/img.width)
            y_height.append(img_height)
            img = cv2.resize(video_img, (width, img_height))
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(img)
            image.append(ImageTk.PhotoImage(img))
            Button.append(tk.Button(window, width = width, image=image[loop], bg="yellow", command=lambda sel=loop: PlayVideo(file_list[sel], player_path), padx=5, pady=5))
            video_img_cv.release()
            
        Button[loop].place(x=0, y=y)
        y_pos.append(y)
        
        y += y_height[loop]+6
        
        loop += 1
        if y > GetSystemMetrics(1):
            break
        
    global check_thread_stop
    
    time.sleep(1)
    
    while True:
        time.sleep(0.1)
        if check_thread_stop == True:
            return
        for idx, btn in enumerate(Button):
            y_pos[idx] -= speed
            btn.place(x=0, y=y_pos[idx])
        
        if y_pos[0] + y_height[0]<= 0:
            del Button[0]
            del y_pos[0]
            gc.collect()
            
        y_max = int(GetSystemMetrics(1))
        
        if y <= y_max+y_height[len(y_height)-1]:
            if loop >= len(file_list):  
                loop = 0
            
            if any((str(file_list[loop])).lower()[file_list[loop].rindex('.')+1:] == ext for ext in check_extension):
                img = Image.open(file_list[loop])
                img_height = int((width*img.size[1])/img.size[0])
                y_height.append(img_height)
                img = img.resize((width, img_height), Image.ANTIALIAS)
                image.append(ImageTk.PhotoImage(img))
                Button.append(tk.Button(window, image=image[loop], bg="white", command=lambda sel=loop: ShowImage(file_list[sel])))
            
                
            else:
                video_img_cv = cv2.VideoCapture(file_list[loop])
                #video_img.open()
                #video_img.open(file_list[loop])
                _, frame = video_img_cv.read()
                #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)  
                #cv2.imshow('test', gray)
                #cv2.imshow('frame', frame)
                #frame = cv2.flip(video_img,1)
                frame_height = frame.shape[0]
                frame_width = frame.shape[1]
                print("frame height: "+str(frame_height))
                print("frame width: "+str(frame_width))
                print("percent: "+str(frame_width/frame_height))
                img_height = int((width*frame_height)/frame_width)
                print("resize width:" + str(width))
                print("resize heigth: "+str(img_height))
                y_height.append(img_height) 
                img = cv2.resize(frame, (width, img_height))
                img = cv2.cvtColor(img,cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(img)
                global player_path
                image.append(ImageTk.PhotoImage(img))
                print(file_list[loop][file_list[loop].rfind("\\")+1:])
                Button.append(tk.Button(window, width = width, image=image[loop], bg="yellow", command=lambda sel=loop: PlayVideo(file_list[sel], player_path), padx=5, pady=5))
                video_img_cv.release()
                #img = cv2.resize(img, (width, img_height))
                
                
            
            #image.append(ImageTk.PhotoImage(img))
        
            
            Button[len(Button)-1].place(x=0, y=y)
            y_pos.append(y)
            loop += 1
            y += img_height+6
            
        y -= speed


    
test = ''
t = ''

def ThreadStop():
    global check_thread_stop
    global test
    global t
    check_thread_stop = True
    test.destroy()
    del t
    
def ThreadStart(root, path, player, width, right, speed):
    global player_path
    player_path = player
    global check_thread_stop
    check_thread_stop = False
    global test
    test = tk.Toplevel(root)
    #test.protocol("WM_DELETE_WINDOW", root.destroy)
    #test.update_idletasks()
    test.overrideredirect(True)
    test.wm_attributes('-topmost', 1)
    if right == True:
        test.geometry(str(width+5)+'x'+str(int(GetSystemMetrics(1)))+'-0+0') #스크롤 화면
    else:
        test.geometry(str(width+5)+'x'+str(int(GetSystemMetrics(1)))+'+0+0') #스크롤 화면
    global t
    t = threading.Thread(target=changeX, args=(test, width, path, right, speed))
    t.daemon = True
    t.start()
    
if __name__=="__main__":
    
    
    root = tk.Tk()
    #root.withdraw()
    root.geometry("150x200+250+250") # 제어화면
    
    root.title("Gallery Scrolling(ver 1.0)")
    #thread
    
    #control gui
    set_path_title = tk.Label(root, text="파일 경로")
    default_path = tk.StringVar()
    set_path = tk.Entry(root, text=default_path)
    default_path.set(".\\files")
    set_player_path_title = tk.Label(root, text="플레이어 경로")
    default_player = tk.StringVar()
    set_player_path = tk.Entry(root, text=default_player)
    default_player.set("C:\\Program Files\\DAUM\\PotPlayer\\PotPlayerMini64.exe")
    set_width_title = tk.Label(root, text="폭 넓이: ")
    default_width = tk.IntVar()
    set_width = tk.Entry(root, width=3, text=default_width)
    default_width.set(200)
    set_right_val = tk.BooleanVar()
    set_right = tk.Checkbutton(root, text="오른쪽에 배치", variable = set_right_val)
    set_speed_title = tk.Label(root, text="속도: ")
    default_speed = tk.IntVar()
    set_speed = tk.Entry(root, width=3, text=default_speed)
    default_speed.set(10)
    set_start = tk.Button(root, text = "시작", command=lambda: ThreadStart(root, default_path.get(), default_player.get(), default_width.get(), set_right_val.get(), default_speed.get()))
    set_stop = tk.Button(root, text = "정지", command=ThreadStop)
    
    set_path_title.grid(row=0, column=0, columnspan=12, sticky=tk.W)
    set_path.grid(row=1, column=0, columnspan=12)
    set_player_path_title.grid(row=2, column=0, columnspan=12, sticky=tk.W)
    set_player_path.grid(row=3, column=0, columnspan=12, sticky=tk.W)
    set_width_title.grid(row=4, column=0, columnspan=4, sticky=tk.E)
    set_width.grid(row=4, column=4, columnspan=8, sticky=tk.W)
    set_right.grid(row=5, column=0, columnspan=12, sticky=tk.W)
    set_speed_title.grid(row=6, column=0, columnspan=2, sticky=tk.W)
    set_speed.grid(row=6, column=2, columnspan=10, sticky=tk.W)
    set_start.grid(row=7, column=0, columnspan=6, sticky=tk.E+tk.W+tk.N+tk.S)
    set_stop.grid(row=7, column=6, columnspan=6, sticky=tk.E+tk.W+tk.N+tk.S)
    
    root.mainloop()
    