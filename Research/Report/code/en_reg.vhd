-- enable logic for register
process(array_reg, en, w_data)
begin
	for i in W+1 downto 0 loop
			array_next(i) <= array_reg(i);
	end loop;

	
	for i in W+1 downto 0 loop
		if(en(i) = '1') then
			array_next(i) <= w_data;
		end if;
	end loop;
end process;
