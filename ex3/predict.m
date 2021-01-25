function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


%(add the bias element)
%add a column of 1's to the first column of X as the same size of X rows.
xlayer = [ones(size(X,1),1),X]; 


%apply the sigmoid function to each element of the Hypothesis. 
%this will give us the next layer (without the bias element). 
layer1 = sigmoid(xlayer * Theta1'); 


%(add the bias element)
%add a column of 1's to the first column of layer1 as the same size of X rows. 
% M size is untouched only, j size changes (we add a collumn for an bias theta)
layer1 = [ones(size(layer1,1),1),layer1]; %

%same as before, but will give us the last layer
layer2 = sigmoid(layer1 * Theta2');

%get the max value per row and tell us the index only (which is the thetas*X highest value)
[values, index] = max(layer2, [] ,2);

%return the highest value INDEX per example (this has a size M)
p = index;



% =========================================================================


end
