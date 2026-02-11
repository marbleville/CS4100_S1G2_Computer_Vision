# Inital proposal

GitHub project link: https://github.com/users/marbleville/projects/3
GitHub repo link: https://github.com/marbleville/CS4100_S1G2_Computer_Vision/tree/main

Topic: Computer vision powered hand gesture video playback control.
Modern media consumption often occurs in contexts where traditional input devices (keyboard, mouse, remote control) are inconvenient or inaccessible, such as controlling a laptop connected to a TV from across a room, or pausing playback while cooking or exercising. While gesture-based control systems exist commercially, they are often proprietary, hardware-dependent, or limited in transparency.
The goal of this project is to design and implement a computer vision–based hand gesture recognition system that enables users to control basic media playback functions (play, pause, volume up/down, next/previous media) using a standard webcam. The project aims to explore foundational computer vision and machine learning techniques by building the recognition pipeline largely from scratch, emphasizing interpretability, feasibility, and real-time usability on commodity hardware.
Gesture recognition has been widely studied in both academic and applied settings. Prior work generally falls into three categories:
Static hand pose recognition, often using convolutional neural networks trained on labeled hand images. Public datasets such as ASL alphabet and digit datasets have been used extensively for classification-based gesture recognition.

Keypoint-based approaches, where hand landmarks are detected and classified using geometric or learned features. Frameworks such as Google’s MediaPipe demonstrate the effectiveness of this approach but abstract away much of the underlying vision pipeline.

Dynamic gesture recognition, which incorporates temporal information to recognize motion-based gestures.
Commercial systems, such as smart TV gesture controls, typically rely on depth sensors or proprietary hardware, while many open-source projects focus narrowly on ASL translation rather than real-time system integration.
Our proposed system prioritizes media control as a practical application, and emphasizes building the core recognition pipeline ourselves, using existing datasets for supervised learning.
The initial system will focus on static hand gestures, allowing reliable recognition using single-frame webcam input. This choice reduces complexity while enabling real-time performance on basic laptops. The pipeline will likely include hand segmentation or detection, feature extraction, and gesture classification using classical or deep learning methods, with architectural choices informed by course material. For our core corpus of gestures, we will make use of existing ASL or other common gestures so that we can use existing labeled datasets for training.
While external libraries for hand detection or landmark extraction may be explored and discussed, the project will be designed such that a fully-from-scratch implementation remains viable if required. The system will be modular, allowing future extension to temporal gestures or user created gestures.
Performance will be evaluated using standard classification metrics (accuracy, confusion matrices) on held-out validation data, as well as qualitative usability testing in real-time scenarios.

# Feedback

Very nice idea, but the core of the project is a single classifier, which will not be sufficient for a 4-person project. I would be interested in motion-based gestures, as that offers some unique challenges. What would be very cool is if you can get this model to run (inference only) on a raspberry pi with an onboard camera - that imposes additional constraints on model architecture and complexity.

A TA will be assigned to each project over the next 1-2 weeks, and will get in touch with you directly. Please use their expertise to your advantage, and feel free to bring any concerns to them as well.

# Feedback Response

Hi Professor,

After reviewing the feedback on our project proposal, we have decided to take both of your suggestions, adding motion gestures and architecting our model to run on a raspberry pi, for our project. We hope this expanded scope will be sufficient for our group.

Another feature idea we had was to incorporate eye tracking to this tool to allow for finer hands-free control. We just wanted to get your take on this idea; it may serve as some additional scope if we are able to complete the rest of our features.

Gradescope submission link: https://www.gradescope.com/courses/1209627/assignments/7529750/submissions/382902185#Question_1-rubric

Thanks in advance,
Laurence, Ethan, Tavishi, and Amartya

Sounds great, thanks Laurence! I do have a raspberry pi with a pi cam in my office that's not being used at the moment; if your team wants, I'd be happy to lend it to you for the semester.

Regards
Raj
