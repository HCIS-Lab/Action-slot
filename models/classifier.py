import torch
import torch.nn as nn

class Head(nn.Module):
	def __init__(self, in_channel, num_ego_classes, num_actor_classes, ego_channel=0):
		super(Head, self).__init__()
		self.num_ego_classes = num_ego_classes
		if self.num_ego_classes !=0:
			if ego_channel == 0:
				self.fc_ego = nn.Sequential(
					nn.ReLU(inplace=False),
					nn.Linear(in_channel, num_ego_classes)
					)
			else:
				self.fc_ego = nn.Sequential(
					nn.ReLU(inplace=False),
					nn.Linear(ego_channel, num_ego_classes)
					)
		self.fc_actor = nn.Sequential(
			nn.ReLU(inplace=False),
			nn.Linear(in_channel, num_actor_classes)
			)

	def forward(self, x, ego_x=None):
		y_actor = self.fc_actor(x)
		if self.num_ego_classes != 0:
			if ego_x != None:
				y_ego = self.fc_ego(ego_x)
			else:
				y_ego = self.fc_ego(x)
		
			return y_ego, y_actor
		else:
			return y_actor

class Instance_Head(nn.Module):
	def __init__(self, in_channel, num_ego_classes, num_actor_classes, ego_channel=0):
		super(Instance_Head, self).__init__()
		self.num_ego_classes = num_ego_classes
		if self.num_ego_classes != 0:
			if ego_channel == 0:
				self.fc_ego = nn.Sequential(
		                nn.ReLU(inplace=False),
		                nn.Linear(in_channel, num_ego_classes),
		                )
			else:
				self.fc_ego = nn.Sequential(
		                nn.ReLU(inplace=False),
		                nn.Linear(ego_channel, num_ego_classes),
		                )
				
		self.fc_actor = nn.ModuleList()
		for i in range(num_actor_classes):
			self.fc_actor.append(nn.Sequential(
	                nn.ReLU(inplace=False),
	                nn.Linear(in_channel, 1),
	                )
				)

	def forward(self, x, ego_x=None):

		b, n, _ = x.shape
		y_actor = []
		for i in range(64):
			y_actor.append(self.fc_actor[i](x[:, i, :]))
		y_actor = torch.stack(y_actor, dim=0)
		y_actor = torch.permute(y_actor, (1, 0, 2))
		y_actor = torch.reshape(y_actor, (b, n))
		# x = torch.reshape(x, (b, n))
		if self.num_ego_classes != 0:
			y_ego = self.fc_ego(ego_x)
			return y_ego, y_actor
		else:
			return y_actor

