tic
close all, clear all, clc
%---------��l��-------------
run = 2000;
step = 1000;
action = 10;%���X�ӧ��
sigma = 1;%�зǮt
epsilon_array = [0 0.1 0.01];%3�ؤ��P��greedy��k
avg_reward_eps = zeros(length(epsilon_array),step);
%----------End-------------
%----------Begin(Example 1, Example 2)-------------
figure; hold on; clr = 'gbr';
for e_i = 1:length(epsilon_array);
    sum_reward = zeros(1, step);
    avg_reward = zeros(1, step);
    opt_arm_cout = zeros(1, step);
    for r_i = 1:run;
        q_star = normrnd(0,sigma,1,action);%�C�@�x�������v   
        N = zeros(1, action);
        Q = zeros(1, action);
        for s_i = 1:step;       
            if(rand(1) <= epsilon_array(e_i))%�P�_�ϥΪ�epsilon
                arm = randi([1,action],1);%�H����@�ӧ��
            else
                [m,arm]=max(Q);
            end
            [m,bestarm] = max(q_star);%�̨θѪ����
            if(arm == bestarm)%�襤OPT
                opt_arm_cout(s_i) = opt_arm_cout(s_i) + 1;%�襤������
            end
            %-------��sQ��(Begin)(method : Incremental Implementation)-----------
            Reward = normrnd(q_star(arm),sigma,1);%�qmean:q*, sigma:1���ͥXreward
            sum_reward(s_i) = sum_reward(s_i) + Reward;%�s�Crun��step1, step2, ...reward
            N(arm) = N(arm) + 1;
            Q(arm) = Q(arm) + (Reward-Q(arm))/N(arm);
            %-------��sQ��(End)-----------
        end    
    end
    avg_reward = sum_reward/run;%������reward
    opt_arm_cout = opt_arm_cout/run;%���ͳ̨θѪ���v  
    avg_reward_eps(e_i,:) = avg_reward;%�x�s������reward
    plot(1:step,opt_arm_cout(1,:),[clr(e_i),'-']);%�e��(example 2)
end
str = {[char(949),'=0.1']};
text(150, 0.78,str,'Color','blue');
str = {[char(949),'=0.01']};
text(750, 0.5,str,'Color','red');
str = {[char(949),'=0(greedy)']};
text(300, 0.2,str,'Color','green');
xlabel( 'Steps' ); ylabel( '% Optimal Action' );
figure; hold on;
%--------------�e��(example 1)-------------
plot(1:step,avg_reward_eps(1,:), 'Color','green');plot(1:step,avg_reward_eps(2,:),'Color','blue');plot(1:step,avg_reward_eps(3,:),'Color','red');
str = {[char(949),'=0.1']};text(100, 1.5,str,'Color','blue');
str = {[char(949),'=0.01']};text(900, 1.2,str,'Color','red');
str = {[char(949),'=0(greedy)']};text(500, 0.7,str,'Color','green');
xlabel( 'Steps' ); ylabel( 'Average Reward' ); 
%--------------End(Example 1, Example 2)------------------
%----------Begin(Example 3)-------------
sum_reward = zeros(1, step);
avg_reward = zeros(1, step);
c=2;%UCB method ���`��
for r_i = 1:run;
    q_star = normrnd(0,sigma,1,action);%�C�@�x�������v
    N = zeros(1, action);
    Q = zeros(1, action);
    Q_UCB = Q;
    for s_i = 1:step;        
        for a_i = 1:action;
            Q_UCB(a_i) = Q(a_i) + c*sqrt(log(s_i)/N(a_i));%�W�[��ɶ����P�_
        end
        [m,arm]=max(Q_UCB);%�A�qQ_UCB�D�̤jaction value��arm
        %-------��sQ��(Begin)(method : Incremental Implementation)-----------
        Reward = normrnd(q_star(arm),sigma,1);
        sum_reward(s_i) = sum_reward(s_i) + Reward;
        N(arm) = N(arm) + 1;
        Q(arm) = Q(arm) + (Reward-Q(arm))/N(arm);
        %-------��sQ��(Begin)-----------
    end     
end
avg_reward = sum_reward/run;
%--------------�e��(Example 3)-------------
figure; hold on;
plot(1:step,avg_reward_eps(2,:),'Color', [0 0 0]+0.05*15);plot(1:step,avg_reward(1,:),'Color','blue');
str = {[char(949),'-greedy ', char(949),'=0.1']};text(600, 1,str,'Color', [0 0 0]+0.05*15);
str = {'UCB c=2'};text(200, 1.6,str,'Color','blue');
xlabel( 'Steps' ); ylabel( 'Average Reward' ); 
%----------End(Example 3)-------------
spend_time = toc