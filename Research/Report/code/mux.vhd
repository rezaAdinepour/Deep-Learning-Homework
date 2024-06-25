-- read multiplexing
with r_addr select
	r_data <= array_reg(0) when "0000000",
					array_reg(1) when "0000001",
					array_reg(2) when "0000010",
					array_reg(3) when "0000011",
					array_reg(4) when "0000100",
					array_reg(5) when "0000101",
					.
					.
					.
					array_reg(127) when others;
