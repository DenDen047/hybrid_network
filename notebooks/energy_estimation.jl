### A Pluto.jl notebook ###
# v0.18.1

using Markdown
using InteractiveUtils

# ╔═╡ 4dfb63f2-d3d1-47ba-a754-8402999bda09
Eₒ = 0.02711e-9

# ╔═╡ ecc6f604-7f60-484b-8aa5-83d7a911a36c
Eᵤ = 0.081e-9

# ╔═╡ feaab574-cde2-4144-9dac-94afa68252a3
Tₛₗ = 25

# ╔═╡ 98b836ea-2d84-4384-be2c-eb782078863d
Δt = 0.0381

# ╔═╡ cfa9d43e-2f3d-4ace-9327-4972fd93e794
S(N, C, fᵢₙ, T) = N * C * fᵢₙ * T * Δt

# ╔═╡ 2efe4501-200e-45c8-a940-5f3cd3b05358
U(N, T) = N * T

# ╔═╡ fc829346-e4d3-11ec-38d5-238e5e291aab
md"# MLP"

# ╔═╡ 68a53b3c-3f94-4620-9fbb-1e64912ad6b9
function mlp(types)
	T = 0
	energy = 0
	for i in 1:length(types)
		layertype = types[i]
		if layertype == "al"
			T = 1
			fᵢₙ = 1 / Δt
		elseif layertype == "sl"
			T = Tₛₗ
			# if i > 1 && types[i-1] == "al"
			# 	fᵢₙ = 1 / Δt
			# else
			# 	fᵢₙ = 0.5
			# end
			fᵢₙ = 0.5
		end

		if i == 1
			N = 784
			C = 500
		elseif i == 2
			N = 500
			C = 500
		elseif i == 3
			N = 500
			C = 10
		end
		
		energy += Eₒ * S(N, C, fᵢₙ, T) + Eᵤ * U(N, T)
	end
	energy += Eᵤ * U(10, T)
	
	return energy
end

# ╔═╡ 97307ef1-8d3a-46ca-950d-063c69059c7b
mlp(["sl" "sl" "sl"])

# ╔═╡ a1d71da1-d50a-45df-8b40-9c75a6312a0a
mlp(["al" "sl" "sl"])

# ╔═╡ 15b4c24a-ac27-4eee-9737-767a9462aac7
mlp(["al" "al" "sl"])

# ╔═╡ 24b25b8f-db09-4542-9634-2c5c79435dd6
mlp(["al" "al" "al"])

# ╔═╡ 5377d168-3719-44c4-83d2-b40c3266072f
md"# CNN"

# ╔═╡ 0b81789a-c11d-4d2d-9370-772eca598f60
function cnn(types)
	T = 0
	energy = 0
	for i in 1:length(types)
		layertype = types[i]
		if layertype == "al"
			T = 1
			fᵢₙ = 1 / Δt
		elseif layertype == "sl"
			T = Tₛₗ
			# if i > 1 && types[i-1] == "al"
			# 	fᵢₙ = 1 / Δt
			# else
			# 	fᵢₙ = 0.5
			# end
			fᵢₙ = 0.5
		end

		if i == 1
			N = 32 * 32 * 3
			C = 3*3 * 32
		elseif i == 2
			N = 30 * 30 * 32
			C = 3*3 * 32
		elseif i == 3
			N = 28 * 28 * 32
			C = 3*3 * 64
		elseif i == 4
			N = 13 * 13 * 64
			C = 3*3 * 64
		elseif i == 5
			N = 5 * 5 * 64
			C = 512
		elseif i == 6
			N = 512
			C = 10
		else
			return -1
		end
		
		energy += Eₒ * S(N, C, fᵢₙ, T) + Eᵤ * U(N, T)
	end
	energy += Eᵤ * U(10, T)
	
	return energy
end

# ╔═╡ e4722096-2128-42d1-ab37-8468946badd8
cnn(["sl" "sl" "sl" "sl" "sl" "sl"])

# ╔═╡ 8a666453-d046-48b2-82c4-d7aa7da1a60e
cnn(["al" "sl" "sl" "sl" "sl" "sl"])

# ╔═╡ c0d0736f-ddab-44c2-81d8-65e6758f600c
cnn(["al" "al" "al" "sl" "sl" "sl"])

# ╔═╡ ffadabb5-bfe1-40ab-9390-0984b52f7824
cnn(["al" "al" "al" "al" "sl" "sl"])

# ╔═╡ 87887cc1-1c07-41b0-89ae-2cc397ab8414
cnn(["al" "al" "al" "al" "al" "al"])

# ╔═╡ Cell order:
# ╠═4dfb63f2-d3d1-47ba-a754-8402999bda09
# ╠═ecc6f604-7f60-484b-8aa5-83d7a911a36c
# ╠═feaab574-cde2-4144-9dac-94afa68252a3
# ╠═98b836ea-2d84-4384-be2c-eb782078863d
# ╠═cfa9d43e-2f3d-4ace-9327-4972fd93e794
# ╠═2efe4501-200e-45c8-a940-5f3cd3b05358
# ╟─fc829346-e4d3-11ec-38d5-238e5e291aab
# ╠═68a53b3c-3f94-4620-9fbb-1e64912ad6b9
# ╠═97307ef1-8d3a-46ca-950d-063c69059c7b
# ╠═a1d71da1-d50a-45df-8b40-9c75a6312a0a
# ╠═15b4c24a-ac27-4eee-9737-767a9462aac7
# ╠═24b25b8f-db09-4542-9634-2c5c79435dd6
# ╟─5377d168-3719-44c4-83d2-b40c3266072f
# ╠═0b81789a-c11d-4d2d-9370-772eca598f60
# ╠═e4722096-2128-42d1-ab37-8468946badd8
# ╠═8a666453-d046-48b2-82c4-d7aa7da1a60e
# ╠═c0d0736f-ddab-44c2-81d8-65e6758f600c
# ╠═ffadabb5-bfe1-40ab-9390-0984b52f7824
# ╠═87887cc1-1c07-41b0-89ae-2cc397ab8414
