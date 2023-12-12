import cv2
import numpy as np

def detect_and_draw_lines(video_source=0):
    # 打开视频捕获对象
    cap = cv2.VideoCapture(video_source)


    while True:
        # 从视频流读取一帧
        ret, frame = cap.read()
        if not ret:
            break

        result_frame = frame.copy()

        # 限制线条只在屏幕中间的一块区域出现
        height, width, _ = result_frame.shape
        roi = result_frame[height//2:height, width//4:3*width//4]

        # 中心点默认坐标
        center = [width // 2, height // 2]

        # 底部中心默认坐标、
        bottom = [width // 2, height]

        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 定义蓝色的HSV范围
        lower_blue = np.array([95, 30, 30])
        upper_blue = np.array([130, 255, 255])

        # 创建蓝色掩码
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # 进行形态学操作，以便更好地检测线条
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel)

        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 初始化变量，用于记录最大面积的轮廓和对应的形状
        max_contour = None
        max_area = 0
        max_shape = ""

        for contour in contours:
            # 计算轮廓的面积
            area = cv2.contourArea(contour)

            # 设置面积阈值来过滤掉噪声
            if area > 80:
                # 如果当前轮廓的面积大于最大面积，则更新最大面积、轮廓和对应的形状
                if area > max_area:
                    max_area = area
                    max_contour = contour

                    # 使用近似多边形拟合轮廓
                    epsilon = 0.03 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)

                    hull = cv2.convexHull(max_contour)
                    
                    # 获取凸包的左上角、右上角和最低点
                    leftmost = tuple(hull[hull[:, :, 0].argmin()][0])
                    rightmost = tuple(hull[hull[:, :, 0].argmax()][0])
                    downmost = tuple(hull[hull[:, :, 1].argmax()][0])
                    topmost = tuple(hull[hull[:, :, 1].argmin()][0])
                    
                    # 调整坐标，加上 ROI 的左上角横坐标和纵坐标
                    leftmost = tuple(np.add(leftmost, [width // 4, height // 2]))
                    rightmost = tuple(np.add(rightmost, [width // 4, height // 2]))
                    downmost = tuple(np.add(downmost, [width // 4, height // 2]))
                    topmost = tuple(np.add(topmost, [width // 4, height //2]))

                    # 根据拟合后的多边形顶点数量判断形状
                    if len(approx) <= 4  and abs(leftmost[0]-rightmost[0])<=100:
                        if abs(topmost[1]-downmost[1])/max(0.0001,abs(leftmost[0]-rightmost[0]))>=6:
                            max_shape = "Straight"
                            center[0] = (topmost[0]+downmost[0])//2
                            center[1] = (leftmost[1]+rightmost[1])//2
                            cv2.circle(result_frame, (center[0],center[1]), 5, (30, 255, 255), 2)  # 黄色点
                            bottom[0] = downmost[0]
                            bottom[1] = downmost[1]
                            cv2.circle(result_frame, (bottom[0],bottom[1]), 5, (30, 255, 255), 2)  # 黄色点
                            
                            degree = np.arctan(center[0]-bottom[0])/(center[1]-bottom[1])*360

                            cv2.putText(result_frame, f"J: {degree}°", (20, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                        else:
                            max_shape = "Stop"
                            center[0] = (topmost[0]+downmost[0])//2
                            center[1] = (leftmost[1]+rightmost[1])//2
                            cv2.circle(result_frame, (center[0],center[1]), 5, (30, 255, 255), 2)  # 黄色点
                            bottom[0] = downmost[0]
                            bottom[1] = downmost[1]
                            cv2.circle(result_frame, (bottom[0],bottom[1]), 5, (30, 255, 255), 2)  # 黄色点
                            
                            degree = np.arctan(center[0]-bottom[0])/max(0.0001,(center[1]-bottom[1]))*360

                            cv2.putText(result_frame, f"J:{480-10-topmost[1]}mm  {degree}°", (20, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


                    elif 5 <= len(approx) < 13:

                        # # 绘制左上角、右上角和最低点、最高点
                        # cv2.circle(result_frame, leftmost, 5, (0, 255, 0), -1)  # 绿色点
                        # cv2.putText(result_frame, f"{leftmost[1]}", (leftmost[0], leftmost[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # cv2.circle(result_frame, rightmost, 5, (150, 60, 100), -1)  # 紫色点
                        # cv2.putText(result_frame, f"{rightmost[1]}", (rightmost[0], rightmost[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 60, 100), 2)
                        
                        # cv2.circle(result_frame, downmost, 5, (0, 0, 0), -1)  # 黑色点
                        # cv2.putText(result_frame, f"{downmost[1]}", (downmost[0], downmost[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                        # cv2.circle(result_frame, topmost, 5, (30, 255, 255), -1)  # 黄色点
                        # cv2.putText(result_frame, f"{topmost[1]}", (topmost[0], topmost[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 255, 255), 2)


                        # 判断十字路口和丁字路口
                        if (abs(leftmost[0]-downmost[0])>55 and abs(rightmost[0]-downmost[0])>55):
                            if abs(topmost[1]-min(rightmost[1],leftmost[1]))>35:
                                max_shape = "Crossroad"
                                center[0] = (topmost[0]+downmost[0])//2
                                center[1] = (leftmost[1]+rightmost[1])//2
                                cv2.circle(result_frame, (center[0],center[1]), 5, (30, 255, 255), 2)  # 黄色点
                                bottom[0] = downmost[0]
                                bottom[1] = downmost[1]
                                cv2.circle(result_frame, (bottom[0],bottom[1]), 5, (30, 255, 255), 2)  # 黄色点

                                degree = np.arctan(center[0]-bottom[0])/max(0.0001,(center[1]-bottom[1]))*360

                                cv2.putText(result_frame, f"J: {(480-center[1])}mm  {degree}°", (20, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                cv2.putText(result_frame, f"L: {degree-90}°", (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                cv2.putText(result_frame, f"R: {degree+90}°", (20, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            else:
                                max_shape = "T-junction"
                                center[0] = (4*downmost[0]+3*(leftmost[0]+rightmost[0]))//10
                                center[1] = (leftmost[1]+rightmost[1])//2
                                cv2.circle(result_frame, (center[0],center[1]), 5, (30, 255, 255), 2)  # 黄色点
                                bottom[0] = downmost[0]
                                bottom[1] = downmost[1]
                                cv2.circle(result_frame, (bottom[0],bottom[1]), 5, (30, 255, 255), 2)  # 黄色点

                                degree = np.arctan(center[0]-bottom[0])/max(0.0001,(center[1]-bottom[1]))*360

                                cv2.putText(result_frame, f"J: {(480-center[1])}mm  {degree}°", (20, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                cv2.putText(result_frame, f"L: {degree-90}°", (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                cv2.putText(result_frame, f"R: {degree+90}°", (20, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                        else:
                        # 判断左转和右转
                            if abs(leftmost[0]-downmost[0]) > abs(rightmost[0]-downmost[0]):
                                if abs(topmost[1]-min(rightmost[1],leftmost[1]))>35:
                                    max_shape = "Left Forward"
                                    center[0] = (topmost[0]+downmost[0])//2
                                    center[1] = (2*leftmost[1]+topmost[1]+downmost[1])//4
                                    cv2.circle(result_frame, (center[0],center[1]), 5, (30, 255, 255), 2)  # 黄色点
                                    bottom[0] = downmost[0]
                                    bottom[1] = downmost[1]
                                    cv2.circle(result_frame, (bottom[0],bottom[1]), 5, (30, 255, 255), 2)  # 黄色点

                                    degree = np.arctan(center[0]-bottom[0])/max(0.0001,(center[1]-bottom[1]))*360

                                    cv2.putText(result_frame, f"J: {(480-center[1])}mm  {degree}°", (20, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                    cv2.putText(result_frame, f"L: {degree-90}°", (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                                else:
                                    max_shape = "Left Turn"
                                    if abs(topmost[0]-leftmost[0])<55 or abs(topmost[0]-downmost[0])>55:
                                        center[0] = (rightmost[0]+downmost[0])//2
                                    else:
                                        center[0] = (topmost[0]+downmost[0])//2
                                    center[1] = (leftmost[1]+4*topmost[1])//5
                                    cv2.circle(result_frame, (center[0],center[1]), 5, (30, 255, 255), 2)  # 黄色点
                                    bottom[0] = downmost[0]
                                    bottom[1] = downmost[1]
                                    cv2.circle(result_frame, (bottom[0],bottom[1]), 5, (30, 255, 255), 2)  # 黄色点

                                    degree = np.arctan(center[0]-bottom[0])/max(0.0001,(center[1]-bottom[1]))*360

                                    cv2.putText(result_frame, f"J: {(480-center[1])}mm  {degree}°", (20, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                    cv2.putText(result_frame, f"L: {degree-90}°", (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


                            elif abs(leftmost[0]-downmost[0]) < abs(rightmost[0]-downmost[0]):
                                if abs(topmost[1]-min(rightmost[1],leftmost[1]))>35:
                                    max_shape = "Right Forward"
                                    center[0] = (topmost[0]+downmost[0])//2
                                    center[1] = (2*rightmost[1]+topmost[1]+downmost[1])//4
                                    cv2.circle(result_frame, (center[0],center[1]), 5, (30, 255, 255), 2)  # 黄色点
                                    bottom[0] = downmost[0]
                                    bottom[1] = downmost[1]
                                    cv2.circle(result_frame, (bottom[0],bottom[1]), 5, (30, 255, 255), 2)  # 黄色点

                                    degree = np.arctan(center[0]-bottom[0])/max(0.0001,(center[1]-bottom[1]))*360

                                    cv2.putText(result_frame, f"J: {(480-center[1])}mm  {degree}°", (20, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                    cv2.putText(result_frame, f"R: {degree+90}°", (20, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                                else:
                                    max_shape = "Right Turn"
                                    if abs(topmost[0]-rightmost[0])<200 or abs(topmost[0]-downmost[0])>200:
                                        center[0] = (leftmost[0]+downmost[0])//2
                                    else:
                                        center[0] = (topmost[0]+downmost[0])//2
                                    center[1] = (rightmost[1]+4*topmost[1])//5
                                    cv2.circle(result_frame, (center[0],center[1]), 5, (30, 255, 255), 2)  # 黄色点
                                    bottom[0] = downmost[0]
                                    bottom[1] = downmost[1]
                                    cv2.circle(result_frame, (bottom[0],bottom[1]), 5, (30, 255, 255), 2)  # 黄色点

                                    degree = np.arctan(center[0]-bottom[0])/max(0.0001,(center[1]-bottom[1]))*360

                                    cv2.putText(result_frame, f"J: {(480-center[1])}mm  {degree}°", (20, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                                    cv2.putText(result_frame, f"R: {degree+90}°", (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    elif len(approx) >= 14:
                        max_shape = "Crossroad"
                        center[0] = (topmost[0]+downmost[0])//2
                        center[1] = (leftmost[1]+rightmost[1])//2
                        cv2.circle(result_frame, (center[0],center[1]), 5, (30, 255, 255), 2)  # 黄色点
                        bottom[0] = downmost[0]
                        bottom[1] = downmost[1]
                        cv2.circle(result_frame, (bottom[0],bottom[1]), 5, (30, 255, 255), 2)  # 黄色点

                        degree = np.arctan(center[0]-bottom[0])/max(0.0001,(center[1]-bottom[1]))*360

                        cv2.putText(result_frame, f"J: {(480-center[1])}mm  {degree}°", (20, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.putText(result_frame, f"L: {degree-90}°", (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.putText(result_frame, f"R: {degree+90}°", (20, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 绘制检测到的蓝色区域
        if max_contour is not None:
            max_contour[:, 0, 0] += width // 4  # 调整 x 坐标，加上 ROI 的左上角横坐标
            max_contour[:, 0, 1] += height // 2  # 调整 y 坐标，加上 ROI 的左上角纵坐标
            cv2.drawContours(result_frame, [max_contour], -1, (0, 255, 0), 1)
            # approx[:, 0, 0] += width // 4  # 调整 x 坐标，加上 ROI 的左上角横坐标
            # approx[:, 0, 1] += height // 2  # 调整 y 坐标，加上 ROI 的左上角纵坐标
            # cv2.drawContours(result_frame, [approx], -1, (60, 30, 30), 1)

        # 在图像中显示识别结果
        cv2.putText(result_frame, f"Main Area Shape: {max_shape}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 显示结果帧q
        cv2.imshow('Result', result_frame)

        # 如果按下 'q' 键，退出视频流窗口
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放视频捕获对象并关闭所有窗口
    cap.release()
    cv2.destroyAllWindows()

# 使用默认相机作为视频源（如果需要，可以更改参数为视频文件路径）
detect_and_draw_lines()
