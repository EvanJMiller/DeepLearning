Homework 3 readme

The first three logic gates (AND, OR, NOT) converged to 100% accuracy 
once back prop was implemented. The training sizes I used were 10, 100, 1000, and 10000
with convergence usually reached by 1000 and sometimes 100 depending on how the weights
were initialized.

The gate that gave me the most trouble was XOR. After fiddling around with both forward and backward
functions, there were times where I still was not able to converge, but achieved an accuracy of about
75%. This biggest issue I had was that the output of the forward pass didn't include a cutoff to
make the output boolean (0 or 1). I then used a .5 boundary and found that the XOR would converge much
more often to 100% accuracy. 

The weights in the output layers were about what I expected and were similar to the ones I found in
homework 2. Understanding what to look for helped debug the first three gates. 

I didn't use the 'Update_Param' function to change my weights, but I did show how it would have been used.
Instead I opted to update each theta as I went through back prop. I also had to cap the limits of the sigmoid
function as I had some weights ballooning to very high values and caused the built in math function to break.

Overall, back prop ended up being very tedious but successful. I found I could change the network sizes of the
XOR gate and achieve slightly higher accuracies when it didn't converge.  