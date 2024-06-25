-- decoding for write address
process(we, w_addr)
begin
	if(we = '0') then
		en <= (others => '0');
	else
	
		case w_addr is
			when "00" => en <= "0001";	
			when "01" => en <= "0010";
			when "10" => en <= "0100";
			when others => en <= "1000";
		end case;
	end if;
end process;
