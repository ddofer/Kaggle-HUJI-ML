library(randomForest)
library (autoencoder)

learn_tree <- read.table("data/train.csv", sep= ',', header=TRUE)
learn_tree$Cover_Type <- as.factor(learn_tree$Cover_Type);
#plot(train$Cover_Type, train$Elevation)
set.seed(1112);
nsamples <- 15120;
sample.learn_tree <- learn_tree[sample(nrow(learn_tree), nsamples), ];

#--------------------------------------------------------------------
# split the dataset into training and test
#--------------------------------------------------------------------
ratio <- 0.8;
train.samples <- ratio*nsamples;
train.rows <- c(sample(nrow(sample.learn_tree), trunc(train.samples)));
train.set  <- sample.learn_tree[train.rows, ];
test.set   <- sample.learn_tree[-train.rows, ];

## Set up the autoencoder architecture:
nl=3 ## number of layers (default is 3: input, hidden, output)
unit.type = "logistic" ## specify the network unit type, i.e., the unit
# N.input = 44 ## number of units (neurons) in the input layer (one unit per pixel)
N.input = 57
N.hidden = 24 ## number of units in the hidden layer

lambda = 0.0005 ## weight decay parameter
beta = 6 ## weight of sparsity penalty term
rho = 0.01 ## desired sparsity parameter
epsilon <- 0.001 ## a small parameter for initialization of weights
## as small gaussian random numbers sampled from N(0,epsilon^2)
max.iterations = 100000 ## number of iterations

cols <- names(train.set)[12:55]

trainmatrix = data.matrix(train.set[,cols], rownames.force = NA)
testmatrix = data.matrix(test.set[,cols], rownames.force = NA)

#cols2 <- names(train.set)[2:55]

autoencoder.object <- autoencode(X.train=trainmatrix,X.test=testmatrix, nl=nl,N.hidden=N.hidden,
                                 unit.type=unit.type,lambda=lambda,beta=beta,rho=rho,epsilon=epsilon,
                                 optim.method="BFGS",max.iterations=max.iterations)
#                                 ,rescale.flag=TRUE,rescaling.offset=0.001)

#output <- predict(autoencoder.object, X.input=trainmatrix, hidden.output=FALSE)$X.output
output2 <- predict(autoencoder.object, X.input=trainmatrix, hidden.output=TRUE)$X.output
output_test <- predict(autoencoder.object, X.input=testmatrix, hidden.output=TRUE)$X.output

cols1 <- names(train.set)[2:11]
train1<-train.set[,cols1]
train1$p1<-output2[,1]
train1$p2<-output2[,2]
train1$p3<-output2[,3]
train1$p4<-output2[,4]

test1<-test.set[,cols1]
test1$p1<-output_test[,1]
test1$p2<-output_test[,2]
test1$p3<-output_test[,3]
test1$p4<-output_test[,4]

set.seed(1);
#tuneRF(train1, train.set$Cover_Type, 1, stepFactor=2, improve=0.05, trace=TRUE)
system.time(clf <- randomForest(train1, train.set$Cover_Type, ntree=1000, mtry=4))
qua = sum(test.set$Cover_Type==predict(clf, test1)) / nrow(test.set)

#qua = sum(train.set$Cover_Type==predict(clf, output2)) / nrow(train.set)

##real testing

test_tree <- read.table("/home/andrew/Projects/kaggle/tree/test.csv", sep= ',', header=TRUE)
testmatrix2 = data.matrix(test_tree[,cols], rownames.force = NA)
output_test2 <- predict(autoencoder.object, X.input=testmatrix2, hidden.output=TRUE)$X.output


test2<-test_tree[,cols1]
test2$p1<-output_test2[,1]
test2$p2<-output_test2[,2]
test2$p3<-output_test2[,3]
test2$p4<-output_test2[,4]

result=predict(clf, test2)

res<-data.frame(test_tree$Id)
names(res)[1] <- "Id"
res$Cover_Type<-as.numeric(result)
write.csv(res, file = "/home/andrew/Projects/kaggle/tree/rf_encoder_result.csv",row.names=FALSE)
