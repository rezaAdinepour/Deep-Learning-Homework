-- decoding for write address
process(we, w_addr)
begin
	if(we = '0') then
		en <= (others => '0');
	else
		for i in 0 to 2**W-1 loop
			if(w_addr = std_logic_vector(to_unsigned(i, w_addr'length))) then
				en(i) <= '1';
			else
				en(i) <= '0';
			end if;
		end loop;
	end if;
end process;
