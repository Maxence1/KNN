import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,y[i]] += - X[i] # 对于分类正确的一列权重参数，对每一个图片，减去n次Xi
        dW[:,j] += X[i]  # 对于分类不正确的几列权重参数，分别加上自己的X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW/= num_train
    
  # Add regularization to the loss.
  loss += 0.5*reg * np.sum(W * W)
  dW += 2*reg*W
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  
  score = np.dot(X,W);
 
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  correct_score_one_colume = score[np.arange(X.shape[0]),y];#取出来正确的分数列向量
  correct_reshape = np.reshape(correct_score_one_colume,[correct_score_one_colume.shape[0],1]);
  score_minus = score - correct_reshape + 1;#原来的分数矩阵减去正确的矩阵  
  score_minus[np.arange(X.shape[0]),y] = 0;#将正确标签的一类置为0，不参与损失计算
  score_minus[score_minus<=0] = 0; # 相当于max函数，去掉了所有负数
  sum_score_minus = np.sum(score_minus,1);#横向求和
  
  sum_s = sum(sum_score_minus);#再纵向求和
  avg_sum = sum_s/X.shape[0];#求平均

  avg_sum_bias = avg_sum + 0.5*reg*np.sum(W*W);
  

  loss+=avg_sum_bias;
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  margin = score_minus;
  margin[margin > 0] = 1.0                         # 示性函数的意义
  row_sum = np.sum(margin, axis=1)                  # 1 by N
  margin[np.arange(X.shape[0]), y] = -row_sum        
  dW += np.dot(X.T, margin)/X.shape[0] + reg * W     # D by C
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
