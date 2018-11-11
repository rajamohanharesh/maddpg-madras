import tensorflow as tf 


with tf.Session() as sess:
	new_list = []
	v = tf.Variable(0)
	new_list.append(v) 
	u = tf.Variable(2)
	new_list.append(u)

	sess.run(tf.global_variables_initializer())
	for var in new_list:
		var = tf.assign(var,var+2)
		var.eval()

	print(sess.run(u))



