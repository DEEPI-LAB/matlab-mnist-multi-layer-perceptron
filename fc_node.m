function parm = fc_node(types, d1, d2)
    
x = strcmp(types,'weight');

if  strcmp(types,'weight')
    parm = 0.1 * randn(d1,d2);
elseif  strcmp(types,'bias')
    parm = 0.1 * randn(d1,1);
else
    error('입력 타입 설정값을 바르게 입력해주세요. ''weight'' 또는 ''bias'' 입니다.')
end
