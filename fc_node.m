function parm = fc_node(types, d1, d2)
    
x = strcmp(types,'weight');

if  strcmp(types,'weight')
    parm = 0.1 * randn(d1,d2);
elseif  strcmp(types,'bias')
    parm = 0.1 * randn(d1,1);
else
    error('�Է� Ÿ�� �������� �ٸ��� �Է����ּ���. ''weight'' �Ǵ� ''bias'' �Դϴ�.')
end
