library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
use IEEE.STD_LOGIC_UNSIGNED.ALL;

entity ram_array is
	generic(	N: integer := 7;     -- 2^7 = 128 word
				M: integer := 32 );  -- 32 bit data
				
	port( clk: in std_logic;
			we: in std_logic; -- write enable
			adr: in std_logic_vector(N - 1 downto 0); -- address
			din: in std_logic_vector(M - 1 downto 0); -- data in
			dout: out std_logic_vector(M - 1 downto 0) ); -- data out
end ram_array;

architecture Behavioral of ram_array is
	type mem_array is array((2 ** N-1) downto 0) of std_logic_vector(M - 1 downto 0); --create 2D array (matrix)
	signal mem: mem_array;

begin
	process(clk)
	begin
		if(rising_edge(clk)) then
			if(we = '1') then
				mem(conv_integer(adr)) <= din; -- write phase
				dout <= (others => 'Z');
			else
				dout <= mem(conv_integer(adr)); -- read phase
			end if;
		end if;
	end process;
end Behavioral;
