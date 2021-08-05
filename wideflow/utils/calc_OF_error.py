import cv2


def calc_error(flow, gt_flow, invalid_mask):

    mag_flow = cv2.sqrt(gt_flow[:, :, 0] * gt_flow[:, :, 0] + gt_flow[:, :, 1] * gt_flow[:, :, 1])


    ret, mask_to_large = cv2.threshold(src=mag_flow, thresh=900, maxval=1, type=cv2.THRESH_BINARY_INV)
    total_inp_mask = invalid_mask[:, :, 0] + invalid_mask[:, :, 1] + invalid_mask[:, :, 2]
    ret, fg_mask = cv2.threshold(src=invalid_mask[:, :, 1], thresh=0.5, maxval=1,
                                        type=cv2.THRESH_BINARY)
    ret, total_mask = cv2.threshold(src=total_inp_mask, thresh=0.5, maxval=1,
                                        type=cv2.THRESH_BINARY)

    # mask_to_large = np.ones(fg_mask.shape)
    bg_mask = total_mask - fg_mask
    ee_base = computeEE(flow, gt_flow)
    result = dict()
    result["FG"] = computer_errors(ee_base, fg_mask * mask_to_large)
    result["BG"] = computer_errors(ee_base, bg_mask * mask_to_large)
    result["Total"] = computer_errors(ee_base, total_mask * mask_to_large)
    return result


def computer_errors(ee_base, mask):
    ee_base = ee_base * mask
    ret, R1 = cv2.threshold(src=ee_base, thresh=1, maxval=1, type=cv2.THRESH_BINARY)
    ret, R2 = cv2.threshold(src=ee_base, thresh=2, maxval=1, type=cv2.THRESH_BINARY)
    ret, R3 = cv2.threshold(src=ee_base, thresh=3, maxval=1, type=cv2.THRESH_BINARY)
    R3 = R3 * mask
    R2 = R2 * mask
    R1 = R1 * mask
    r3_sum = cv2.sumElems(R3)[0]
    r2_sum = cv2.sumElems(R2)[0]
    r1_sum = cv2.sumElems(R1)[0]
    ee = cv2.sumElems(ee_base)[0]
    no_p = cv2.sumElems(mask)[0]
    result = {"ee" : ee, "R1": r1_sum, "R2": r2_sum, "R3" : r3_sum, "noPoints" : no_p}
    return result


def computeEE(src0, src1):
    diff_flow = src0 - src1
    res = (diff_flow[:, :, 0] * diff_flow[:, :, 0]) + (diff_flow[:, :, 1] * diff_flow[:, :, 1])
    return cv2.sqrt(res)