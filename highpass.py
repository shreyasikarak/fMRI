import cv2
import matplotlib.pyplot as plt
import os
import pdb

def main():	

        path= "/home/silp150/shreyashi/100307/unprocessed/3T/tfMRI_EMOTION_RL/"
        
        '''os.chdir(path)
        img= cv2.imread('image3.png',1)'''

        img=cv2.imread(os.path.join(path,'image.png'),1)
        img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        edges= cv2.Laplacian(img,-1,ksize=31,
                scale=1, delta=0, borderType= cv2.BORDER_DEFAULT)
        output=[img,edges]
        titles=['original',' Edges']

        for i in range(2):
                plt.subplot(1,2, i+1)
                plt.imshow(output[i], cmap='gray')
                plt.title(titles[i])
                plt.xticks([])
                plt.yticks([])

        plt.show()
        pdb.set_trace()
        cv2.waitkey(0)

if __name__ == "__main__":
        main()
