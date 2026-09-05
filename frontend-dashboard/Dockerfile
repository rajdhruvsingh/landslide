# --- Build toolchain (shared deps for both targets) ---
FROM node:20-alpine AS base
WORKDIR /app
COPY package*.json ./
RUN npm ci

# --- Dev target: Vite dev server (used by infra/docker-compose.yml) ---
FROM base AS dev
COPY . .
EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]

# --- Build target: static bundle ---
FROM base AS build
COPY . .
RUN npm run build

# --- Prod target: static assets behind nginx ---
FROM nginx:alpine AS prod
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80